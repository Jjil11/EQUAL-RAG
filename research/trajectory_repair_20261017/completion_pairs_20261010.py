"""Training-only counterfactual completion compiler; no held-out inputs.

Gold evidence edges are training labels, never inputs to the inference policy.
Citation support here is lexical/local, not a semantic entailment certificate.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import hashlib
import json
import re
import string
from pathlib import Path

ROOT = Path(str(_release_root))
ROUND = ROOT / 'runs/completion_repair_20261010'


def norm(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def units(text):
    # Training unit boundaries are closed citation groups, not commas inside
    # proper names (e.g. "Hetland, Jr. [3]"). Dangling text is not a unit.
    return [(m.start(1), m.end(1), m.group(1).strip()) for m in
            re.finditer(r'(?:^|,\s*)(.*?\[\d+\](?:\s*\[\d+\])*)', text)
            if norm(m.group(1))]


def true_repeat(row):
    seen=set()
    for lo,hi,unit in units(row['short']):
        value=norm(unit)
        if value in seen:
            return row['short'][:lo].rstrip().rstrip(',').rstrip(),unit
        seen.add(value)
    return None


def valid_training_edges(edges):
    # The source contains empty official answer aliases. They are not a fact
    # to complete, and must never become a synthetic " [doc]" training target.
    return [e for e in edges if norm(e['answer']) and any(norm(a) for a in e['aliases'])]


def edge_matches(unit, edges):
    value = norm(unit)
    refs = {int(x) for x in re.findall(r'\[(\d+)\]', unit)}
    return {i for i, e in enumerate(edges) if value
            if value in {norm(a) for a in e['aliases']}
            and refs.intersection(e.get('supporting_doc_ids', [e['doc_id']]))}


def covered(text, edges):
    return set().union(*(edge_matches(u, edges) for _, _, u in units(text)))


def repair(row):
    prefix = row['trajectory_prefix'].rstrip().rstrip(',').rstrip()
    seen = covered(prefix, row['edges'])
    missing = [i for i in range(len(row['edges'])) if i not in seen]
    rendered = [f"{row['edges'][i]['answer']} [{row['edges'][i]['doc_id']}]" for i in missing]
    chosen = prefix + (', ' + ', '.join(rendered) if rendered else '')
    rejected = row['trajectory_prefix'] + row['trajectory_continuation']
    if chosen == rejected or covered(chosen, row['edges']) != set(range(len(row['edges']))):
        raise ValueError(f"Invalid completion label {row['id']}")
    return {'chosen': chosen, 'rejected': rejected, 'prefix': prefix,
            'missing_edges': missing, 'action': 'complete_then_stop' if missing else 'stop'}


def encode_end(tokenizer, prompt, response, stops):
    import torch
    full = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}],
        tokenize=False, add_generation_prompt=False,
        enable_thinking=not tokenizer._equal_rag_disable_thinking)
    start = full.rfind(response)
    if start < 0:
        raise ValueError('Response absent from template')
    enc = tokenizer(full, return_offsets_mapping=True, truncation=False)
    end = next(i for i, (token, offset) in enumerate(zip(enc['input_ids'], enc['offset_mapping']))
               if token in stops and offset[0] >= start)
    # Include exactly the true assistant EOS; exclude post-EOS suffix.
    return torch.tensor(enc['input_ids'][:end+1]), enc['offset_mapping'][:end+1], start


def compiled_pair(tokenizer, row, stops, max_length=2048):
    spec = repair(row)
    chosen, offsets, start = encode_end(tokenizer, row['prompt'], spec['chosen'], stops)
    rejected, _, _ = encode_end(tokenizer, row['prompt'], spec['rejected'], stops)
    boundary = start + len(spec['prefix'])
    # Do not retain a token merging a closed citation with a continuation comma.
    # Offset counts can exclude special tokens: obtain the positional limit.
    limit = next((i for i, (lo, hi) in enumerate(offsets) if hi > boundary), len(chosen)-1)
    shared = 0
    for a, b in zip(chosen.tolist(), rejected.tolist()):
        if a != b or shared >= limit:
            break
        shared += 1
    if not 0 < shared < min(len(chosen), len(rejected)) or max(len(chosen),len(rejected)) > max_length:
        raise ValueError(f"Empty or truncated branch {row['id']}")
    assert chosen[:shared].tolist() == rejected[:shared].tolist()
    assert int(chosen[-1]) in stops and int(rejected[-1]) in stops
    spec.update(shared_tokens=shared, chosen_tokens=len(chosen)-shared,
                rejected_tokens=len(rejected)-shared,
                first_chosen_token=int(chosen[shared]), first_rejected_token=int(rejected[shared]))
    return chosen, rejected, shared, spec


def retention_mask(tokenizer, row, max_length=2048):
    import torch
    import train_trajectory_equal as legacy
    ids, _, labels = legacy.encode_example(tokenizer, row['prompt'], row['short'], max_length)
    full = tokenizer.apply_chat_template(
        [{'role':'user','content':row['prompt']},{'role':'assistant','content':row['short']}],
        tokenize=False, add_generation_prompt=False,
        enable_thinking=not tokenizer._equal_rag_disable_thinking)
    start = full.rfind(row['short'])
    enc = tokenizer(full, return_offsets_mapping=True, truncation=True, max_length=max_length)
    if enc['input_ids'] != ids.tolist() or start < 0:
        raise ValueError('KL position alignment mismatch')
    spans, seen = [], set()
    for lo, hi, u in units(row['short']):
        matches = edge_matches(u, row['edges'])
        if matches and not matches.intersection(seen):
            spans.append((start+lo, start+hi))
            seen.update(matches)
    # Teacher positions after repeats may be useful. Keep unique supported units
    # wherever they occur, not just tokens before the first repetition.
    positions = (labels[1:] != -100).nonzero().flatten().tolist()
    mask = [any(lo >= a and hi <= b and hi > lo for a,b in spans)
            for p in positions for lo,hi in [enc['offset_mapping'][p+1]]]
    return torch.tensor(mask, dtype=torch.bool)


def main():
    from transformers import AutoTokenizer, GenerationConfig
    parser=argparse.ArgumentParser(); parser.add_argument('--family',required=True,choices=['qwen3','mistral'])
    args=parser.parse_args(); family=args.family
    filename='train_trajectory_qwen3_14b_onpolicy_seed20260930.jsonl' if family=='qwen3' else 'train_trajectory_onpolicy_seed20260930.jsonl'
    source=ROOT/'data/qampari_evidence_full'/filename
    model=str(_release_model) if family=='qwen3' else '/common_data/model/Mistral-7B-Instruct-v0.3'
    tok=AutoTokenizer.from_pretrained(model,use_fast=True); tok._equal_rag_disable_thinking=family=='qwen3'
    stop=GenerationConfig.from_pretrained(model).eos_token_id; stops=[stop] if isinstance(stop,int) else list(stop)
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    audit=[]
    for row in rows:
        original_edge_count=len(row['edges'])
        row['edges']=valid_training_edges(row['edges'])
        removed_edges=original_edge_count-len(row['edges'])
        original_active=bool(row.get('has_trajectory'))
        row['original_has_trajectory']=original_active
        if original_active:
            repeated=true_repeat(row)
            if repeated is None or not row['edges']:
                row['has_trajectory']=False
            else:
                prefix,unit=repeated
                row['trajectory_prefix']=prefix
                row['trajectory_continuation']=', '+unit
        mask=retention_mask(tok,row)
        row['retention_mask']=mask.tolist()
        record={'id':row['id'],'retained_kl_tokens':int(mask.sum()),'original_kl_tokens':len(mask),
                'original_active':original_active,'verified_repeat':bool(row.get('has_trajectory')),
                'invalid_empty_training_edges_removed':removed_edges}
        if row.get('has_trajectory'):
            _,_,_,spec=compiled_pair(tok,row,stops)
            row['completion_pair']=spec
            record.update(spec)
        audit.append(record)
    out=ROUND/'compiled_v3';out.mkdir(parents=True,exist_ok=True)
    target=out/f'{family}_train.jsonl'
    if target.exists(): raise FileExistsError('Refusing to overwrite compiled training labels')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    active=[r for r in rows if r.get('has_trajectory')]
    stop_rows=[r for r in active if r['completion_pair']['action']=='stop'][:4]
    continue_rows=[r for r in active if r['completion_pair']['action']=='complete_then_stop'][:8-len(stop_rows)]
    (out/f'{family}_smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in stop_rows+continue_rows))
    report={'family':family,'rows':len(rows),'active_pairs':sum('action' in x for x in audit),
            'original_active':sum(x['original_active'] for x in audit),
            'false_or_unverified_repeat_labels_removed':sum(x['original_active'] and not x['verified_repeat'] for x in audit),
            'invalid_empty_training_edges_removed':sum(x['invalid_empty_training_edges_removed'] for x in audit),
            'stop_pairs':sum(x.get('action')=='stop' for x in audit),
            'completion_pairs':sum(x.get('action')=='complete_then_stop' for x in audit),
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'compiled_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'retained_kl_tokens':sum(x['retained_kl_tokens'] for x in audit),
            'original_kl_tokens':sum(x['original_kl_tokens'] for x in audit),
            'rows_with_no_retained_tokens':sum(x['retained_kl_tokens']==0 for x in audit),
            'heldout_examples_read':0,'per_row':audit}
    (out/f'{family}_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)


if __name__=='__main__':main()

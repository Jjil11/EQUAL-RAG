"""Deterministic tokenizer-only audit; mismatched BPE is NOT proof of impossibility."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from suite_common import SUITE, PROJECT, dump, rows, read, records, sha

CANDIDATES = [
    ('Qwen/Qwen3-14B', 'Qwen byte-level BPE'),
    ('Qwen/Qwen2.5-7B-Instruct', 'Qwen byte-level BPE'),
    ('meta-llama/Llama-3.1-8B-Instruct', 'Llama-3 byte-level BPE'),
    ('mistralai/Mistral-7B-Instruct-v0.3', 'Mistral SentencePiece BPE'),
    ('google/gemma-2-9b-it', 'Gemma SentencePiece'),
    ('microsoft/Phi-3-mini-4k-instruct', 'Llama-2/SentencePiece'),
    ('openai-community/gpt2', 'GPT-2 byte-level BPE'),
]
FILES = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'added_tokens.json', 'tokenizer.model', 'tokenizer.model.v3', 'vocab.json',
         'merges.txt', 'config.json', 'chat_template.jinja']

def text_pairs():
    from entity_abc_contracts_20261015 import parse_closed
    out = []
    for r in records():
        if r['type'] != 'A':
            continue
        parsed = parse_closed(r['positive_text'])
        assert parsed and r['past_unit_count'] > 0
        end = parsed[0][r['past_unit_count'] - 1]['end']
        native, remainder = r['positive_text'][:end], r['positive_text'][end:]
        assert native.endswith(']') and remainder.startswith(', ')
        assert r['actual_ids'][-2:] == [60, 151645]
        out.append({'id': r['id'], 'native': native, 'rolled': native[:-1],
                    'span': remainder[2:], 'joint': r['positive_text']})
        assert native[:-1] + '], ' + remainder[2:] == r['positive_text']
    assert len(out) == 21
    return out

def audit(name, family, encode, decode, provenance, pairs):
    one = {s: encode(s) for s in [']', '],', '], ']}
    merged = sorted({i for s in ['],', '], '] for i in one[s]
                     if decode([i]) in ['],', '], '] and i not in one[']']})
    rr = []
    for p in pairs:
        separate = encode(p['native']) + encode(', ' + p['span'])
        natural = encode(p['joint'])
        rolled_joint = encode(p['rolled'] + '], ' + p['span'])
        assert natural == rolled_joint
        k = next((i for i, (a, b) in enumerate(zip(separate, natural)) if a != b), min(len(separate), len(natural)))
        rr.append({'id': p['id'], 'mismatch': separate != natural,
                   'first_different_token': k, 'naive_ids': separate, 'joint_ids': natural,
                   'naive_boundary_window': separate[max(0, k-4):k+10],
                   'joint_boundary_window': natural[max(0, k-4):k+10]})
    result = {'tokenizer': name, 'family': family, 'provenance': provenance,
              'literal_ids': one, 'decoded_literal_pieces': {s: [decode([i]) for i in one[s]] for s in one},
              'distinct_merged_continuation': bool(merged), 'merged_continuation_ids': merged,
              'records': 21, 'mismatches': sum(x['mismatch'] for x in rr),
              'boundary_mismatch_rate_percent': 100*sum(x['mismatch'] for x in rr)/21,
              'context_native_example_ids': encode('Alice [1], Bob [1]'),
              'context_continuation_example_ids': encode('Alice [1], Bob [1], Carol [2]'),
              'details': rr, 'not_an_inference_deadlock_proof': True}
    print(json.dumps({k:v for k,v in result.items() if k not in ['details','provenance']}, ensure_ascii=False), flush=True)
    return result

def run(online=False, output_subdir='analysis'):
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi, snapshot_download
    pairs = text_pairs()
    target = SUITE / output_subdir
    results, failures = [], []
    pair_path = SUITE / 'data/state_a_text_pairs.jsonl'
    if pair_path.exists():
        assert read(pair_path) == pairs
    else:
        rows(pair_path, pairs)
    candidates = [(name, family, None) for name, family in CANDIDATES]
    local = PROJECT / 'artifacts/equal_rag_cequal_dev_calibration_20261002/runs/cequal_dev_calibration/c1_quality_mid/adapter'
    if (local / 'tokenizer.json').exists():
        candidates.append(('local:Mistral-7B-Instruct-v0.3', 'Mistral SentencePiece BPE', local))
    for name, family, local_path in candidates:
        try:
            provenance = {'repo': name, 'network_used': False}
            if local_path:
                folder = str(local_path)
            elif online:
                info = HfApi(token=False).model_info(name, timeout=15)
                folder = snapshot_download(name, revision=info.sha, allow_patterns=FILES,
                                           cache_dir=str(SUITE/'tokenizers/hf_cache'), max_workers=2, token=False)
                provenance.update(revision=info.sha, network_used=True)
            else:
                folder = name
            tok = AutoTokenizer.from_pretrained(folder, use_fast=True, local_files_only=True, trust_remote_code=False)
            provenance['class'] = tok.__class__.__name__
            provenance['files'] = {p.name:sha(p) for p in Path(folder).iterdir() if p.is_file()} if Path(folder).is_dir() else {}
            model = json.loads(tok.backend_tokenizer.to_str())['model']
            provenance['tokenizer_model_type'] = model['type']
            provenance['vocabulary_and_merges_sha256'] = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
            if name == 'Qwen/Qwen3-14B':
                for r, p in zip([v for v in records() if v['type']=='A'], pairs):
                    assert tok.decode(r['actual_ids'][r['prompt_tokens']:-1], skip_special_tokens=False) == p['native']
                    assert tok.decode(r['positive_ids'][r['prompt_tokens']:-1], skip_special_tokens=False) == p['joint']
                provenance['all_21_reconstructed_native_and_positive_texts_match_frozen_token_ids'] = True
            results.append(audit(name, family, lambda s:tok.encode(s, add_special_tokens=False),
                                 lambda ids:tok.decode(ids, skip_special_tokens=False), provenance, pairs))
        except Exception as e:
            failures.append({'tokenizer':name,'error_type':type(e).__name__,'error':str(e)[:700]})
            print('UNAVAILABLE '+name+' '+type(e).__name__, flush=True)
    try:
        import tiktoken
        for name in ['cl100k_base', 'o200k_base']:
            try:
                enc = tiktoken.get_encoding(name)
                provenance = {'library':'tiktoken','version':importlib.metadata.version('tiktoken'),
                              'explicit_tokenizer_only_comparator_not_a_second_backbone':True}
                results.append(audit(name, 'tiktoken '+name, enc.encode, enc.decode, provenance, pairs))
            except Exception as e:
                failures.append({'tokenizer':name,'error_type':type(e).__name__,'error':str(e)[:700]})
    except ImportError:
        failures.append({'tokenizer':'tiktoken','error':'not installed'})
    out = {'status':'COMPLETE' if results else 'NO_TOKENIZERS_AVAILABLE', 'results':results,'unavailable':failures,
           'records_sha256':sha(SUITE/'data/state_a_text_pairs.jsonl'), 'frozen_records_sha256':sha(PROJECT/'artifacts/qwen_pair_aonly_20261015/verified/run/compiled/records.jsonl'),
           'script_sha256':sha(__file__), 'interpretation':'Deterministic segmentation mismatch, not measured model inability, universal deadlock, or cross-model effectiveness.',
           'successful_tokenizers':len(results), 'distinct_declared_families':len({r['family'] for r in results})}
    dump(target/'tokenizer_boundary_audit.json',out)
    table = ['# Table S1: tokenizer boundary audit','', '| Tokenizer | Family | `]` IDs | `],` IDs | `], ` IDs | Merged | Mismatch / 21 | Rate |',
             '|---|---|---|---|---|---|---|---|']
    for r in results:
        ids=r['literal_ids']; table.append(f"| {r['tokenizer']} | {r['family']} | {ids[']']} | {ids['],']} | {ids['], ']} | {r['distinct_merged_continuation']} | {r['mismatches']}/21 | {r['boundary_mismatch_rate_percent']:.2f}% |")
    table += ['', 'Unavailable tokenizers are explicitly logged, not counted as zero mismatch.', '', out['interpretation']]
    from suite_common import safe
    with safe(target/'TABLE_S1.md').open('x',encoding='utf-8') as f:f.write('\n'.join(table)+'\n')
    return out

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--online',action='store_true');ap.add_argument('--output-subdir',default='analysis');a=ap.parse_args()
    run(a.online,a.output_subdir)

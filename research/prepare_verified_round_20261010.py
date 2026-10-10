"""Freeze prospective non-oracle retrieval holdout and check all train boundaries."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(str(_release_root))
sys.path.insert(0, str(ROOT/'code'))
from train_verified_boundary_20261010 import boundary_ids, aligned_trajectory
from build_qampari_pilot import clean, prompt_for


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, rows):
    Path(path).write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows))


def main():
    report = {}
    for family, model, filename in [
        ('qwen3', str(_release_model), 'train_trajectory_qwen3_14b_onpolicy_seed20260930.jsonl'),
        ('mistral', '/common_data/model/Mistral-7B-Instruct-v0.3', 'train_trajectory_onpolicy_seed20260930.jsonl')]:
        tok = AutoTokenizer.from_pretrained(model)
        tok._equal_rag_disable_thinking = family == 'qwen3'
        rows = read(ROOT/'data/qampari_evidence_full'/filename)
        active = [x for x in rows if x.get('has_trajectory')]
        target_ids, errors, merged = set(), [], 0
        for row in active:
            try:
                ids, both, n, stop, _, crossed = aligned_trajectory(tok,row,2048)
                teacher, _, _, _ = boundary_ids(tok,row['prompt'],row['short'],2048)
                assert teacher[:n].tolist() == ids.tolist(), 'Retokenized teacher prefix differs'
                merged += int(crossed > 0)
                target_ids.add(stop)
            except Exception as exc:
                errors.append({'id':row['id'],'error':str(exc)})
        report[family] = {'rows':len(rows),'active':len(active),'eot_ids':sorted(target_ids),
                          'boundary_crossing_bpe_tokens':merged,'errors':errors}
        if errors:
            raise ValueError(json.dumps(report))
        smoke = active[:8] + [x for x in rows if not x.get('has_trajectory')][:8]
        dump(ROOT/f'runs/submission_20261010/smoke_{family}.jsonl', smoke)
    seen_ids, seen_q = set(), set()
    for p in (ROOT/'data').rglob('*.jsonl'):
        if 'qampari' not in str(p).lower():
            continue
        for row in read(p):
            seen_ids.add(str(row.get('id','')))
            seen_q.add(' '.join(row.get('question','').lower().split()))
    for p in (ROOT/'runs').rglob('eval_results.jsonl'):
        for row in read(p):
            seen_ids.add(str(row.get('id','')))
    source = ROOT/'ALCE/data/qampari_eval_gtr_top100.json'
    raw = json.loads(source.read_text())
    rows = []
    for i, item in enumerate(raw):
        ident = str(item.get('id',item.get('sample_id',i)))
        q = ' '.join(item['question'].lower().split())
        if ident in seen_ids or q in seen_q:
            continue
        docs = item['docs'][:5]  # Strict retrieval rank: never use gold to choose docs.
        supports = {}
        for j,d in enumerate(docs,1):
            for k,present in enumerate(d.get('answers_found') or []):
                if present:
                    supports.setdefault(k,[]).append(j)
        edges = [{'answer_idx':k,'answer':clean(item['answers'][k][0]),
                  'aliases':[clean(a) for a in item['answers'][k]],
                  'doc_id':v[0],'supporting_doc_ids':v} for k,v in sorted(supports.items())
                 if k < len(item['answers']) and item['answers'][k]]
        rows.append({'id':ident,'question':clean(item['question']),
                     'prompt':prompt_for(item['question'],docs),
                     'docs':[{'id':j,'title':clean(d.get('title','')),'text':clean(d.get('text',''))}
                             for j,d in enumerate(docs,1)],
                     'edges':edges,'all_gold_answers':item['answers'],
                     'short':', '.join(f"{e['answer']} [{e['doc_id']}]" for e in edges)+'.',
                     'retrieval_setting':'unaltered_gtr_top5'})
    rows.sort(key=lambda x:hashlib.sha256(('prospective-20261010:'+x['id']).encode()).hexdigest())
    out = ROOT/'data/prospective_qampari_gtr5_20261010'
    if out.exists():
        raise FileExistsError('Prospective set already frozen; never overwrite it')
    out.mkdir()
    dump(out/'dev.jsonl',rows[:80])
    dump(out/'holdout.jsonl',rows[80:])
    manifest = {'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'n':len(rows),'dev':len(rows[:80]),'holdout':len(rows[80:]),
                'exclude_materialized_ids_and_normalized_questions':True,
                'retrieval_gold_selection':False,'top_k':5,
                'no_model_output_selection':True,'no_min_answer_filter':True,
                'zero_supported_answers':sum(not x['edges'] for x in rows),
                'mean_supported_answers':sum(len(x['edges']) for x in rows)/max(1,len(rows)),
                'warning':'Prospective controlled split of unused ALCE QAMPARI evaluation rows; NOT an official benchmark or an external dataset.',
                'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')}}
    tok = AutoTokenizer.from_pretrained(str(_release_model))
    lengths=[len(tok(tok.apply_chat_template([{'role':'user','content':x['prompt']}],tokenize=False,
                     add_generation_prompt=True,enable_thinking=False))['input_ids']) for x in rows]
    manifest['max_qwen_prompt_tokens']=max(lengths)
    manifest['prompts_over_2048']=sum(n>2048 for n in lengths)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report['prospective']=manifest
    (ROOT/'runs/submission_20261010/preflight.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()

"""Repair missing raw-retrieval coverage annotations without changing frozen IDs."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
import re
import string
from pathlib import Path

ROOT=Path(str(_release_root))


def norm(s):
    s=''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())


def main():
    old=ROOT/'data/prospective_qampari_gtr5_20261010'
    out=ROOT/'data/prospective_qampari_gtr5_v2_20261010'
    if out.exists(): raise FileExistsError('Never overwrite a frozen dataset')
    out.mkdir()
    report={'v1_invalid_reason':'Raw GTR files lack answers_found; v1 edges are invalid and must not be evaluated.',
            'same_ids_questions_docs_prompts_and_split_as_v1':True,
            'matching':'ALCE-normalized alias substring in retrieved title+text; lexical proxy, not entailment.',
            'no_model_prediction_used':True,'splits':{}}
    for split in ['dev','holdout']:
        rows=[json.loads(x) for x in (old/f'{split}.jsonl').read_text().splitlines()]
        for row in rows:
            supports={}
            for doc in row['docs']:
                text=norm(doc['title']+' '+doc['text'])
                for idx,aliases in enumerate(row['all_gold_answers']):
                    if any(norm(a) and norm(a) in text for a in aliases):
                        supports.setdefault(idx,[]).append(doc['id'])
            row['edges']=[{'answer_idx':idx,'answer':row['all_gold_answers'][idx][0],
                          'aliases':row['all_gold_answers'][idx],'doc_id':refs[0],
                          'supporting_doc_ids':refs} for idx,refs in sorted(supports.items())]
            row['short']=', '.join(f"{e['answer']} [{e['doc_id']}]" for e in row['edges'])+'.'
        path=out/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        report['splits'][split]={'n':len(rows),'zero_supported_answers':sum(not x['edges'] for x in rows),
                               'mean_supported_answers':sum(len(x['edges']) for x in rows)/len(rows),
                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report['v1_manifest_sha256']=hashlib.sha256((old/'manifest.json').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    (old/'DO_NOT_USE_INVALID_COVERAGE.json').write_text(json.dumps({'invalid':True,'replacement':str(out),'reason':report['v1_invalid_reason']},indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

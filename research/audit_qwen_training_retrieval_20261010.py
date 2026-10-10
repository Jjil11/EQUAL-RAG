"""Train-ID-only retrieval condition audit; zero lexical edges != no evidence."""
import hashlib
import json
from pathlib import Path
from completion_pairs_20261010 import ROOT,valid_training_edges


def main():
    trainfile=ROOT/'runs/completion_repair_20261010/compiled_v3/qwen3_train.jsonl'
    source=ROOT/'ALCE/data/qampari_eval_gtr_top100.json'
    train=[json.loads(x) for x in trainfile.read_text().splitlines() if x.strip()]
    ids={r['id'] for r in train};assert len(ids)==len(train)==350
    raw={str(r.get('id',r.get('sample_id',i))):r for i,r in enumerate(json.loads(source.read_text()))
         if str(r.get('id',r.get('sample_id',i))) in ids}
    records=[]
    for row in train:
        item=raw.get(row['id'])
        if item is None:
            records.append({'id':row['id'],'source_found':False});continue
        supports={}
        for j,doc in enumerate(item['docs'][:5],1):
            for k,present in enumerate(doc.get('answers_found') or []):
                if present and k<len(item['answers']) and item['answers'][k]:supports.setdefault(k,[]).append(j)
        edges=[{'answer':item['answers'][k][0],'aliases':item['answers'][k],'doc_id':docs[0],
                'supporting_doc_ids':docs} for k,docs in supports.items()]
        clean=valid_training_edges(edges)
        old=valid_training_edges(row['edges'])
        records.append({'id':row['id'],'source_found':True,'existing_lexical_edges':len(old),
            'unaltered_top5_lexical_edges':len(clean),
            'existing_zero_lexical_support':not old,'top5_zero_lexical_support':not clean})
    found=[r for r in records if r['source_found']]
    report={'status':'COMPLETE_TRAIN_ONLY_AUDIT','train_rows':350,'source_joined_rows':len(found),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'train_sha256':hashlib.sha256(trainfile.read_bytes()).hexdigest(),
        'existing_zero_lexical_support':sum(r['existing_zero_lexical_support'] for r in found),
        'top5_zero_lexical_support':sum(r['top5_zero_lexical_support'] for r in found),
        'existing_lexical_edges':sum(r['existing_lexical_edges'] for r in found),
        'top5_lexical_edges':sum(r['unaltered_top5_lexical_edges'] for r in found),
        'heldout_rows_analyzed':0,'model_generations':0,
        'warning':'Official raw source file contains other rows, but only the 350 existing training IDs are analyzed. Zero lexical support is not proof of semantic evidence absence and must not automatically become an abstention gold label.',
        'per_row':records}
    out=ROOT/'runs/qwen_training_retrieval_audit_20261010';out.mkdir(exist_ok=True)
    target=out/'summary.json'
    if target.exists():raise FileExistsError('Audit already frozen')
    target.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)


if __name__=='__main__':main()

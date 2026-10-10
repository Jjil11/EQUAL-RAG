"""Correct train-only coverage audit using the frozen v2 lexical definition."""
import hashlib
import json
import re
import string
from completion_pairs_20261010 import ROOT,valid_training_edges


def norm(text):
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def edges_for(item):
    supports={}
    for j,doc in enumerate(item['docs'][:5],1):
        text=norm(doc.get('title','')+' '+doc.get('text',''))
        for i,aliases in enumerate(item['answers']):
            if aliases and any(norm(a) and norm(a) in text for a in aliases):supports.setdefault(i,[]).append(j)
    return [{'answer':item['answers'][i][0],'aliases':item['answers'][i],
             'doc_id':docs[0],'supporting_doc_ids':docs} for i,docs in sorted(supports.items())]


def main():
    source=ROOT/'ALCE/data/qampari_eval_gtr_top100.json'
    trainfile=ROOT/'runs/completion_repair_20261010/compiled_v3/qwen3_train.jsonl'
    train=[json.loads(x) for x in trainfile.read_text().splitlines() if x.strip()]
    ids={r['id'] for r in train};assert len(ids)==len(train)==350
    raw={str(r.get('id',r.get('sample_id',i))):r for i,r in enumerate(json.loads(source.read_text()))
         if str(r.get('id',r.get('sample_id',i))) in ids}
    assert set(raw)==ids
    records=[]
    for row in train:
        item=raw[row['id']];assert norm(row['question'])==norm(item['question'])
        new=valid_training_edges(edges_for(item));old=valid_training_edges(row['edges'])
        records.append({'id':row['id'],'existing_edges':len(old),'raw_top5_edges':len(new),
                        'existing_zero':not old,'raw_top5_zero':not new})
    report={'status':'COMPLETE_CORRECTED_TRAIN_ONLY_AUDIT','train_rows':350,'source_joined_rows':350,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'train_sha256':hashlib.sha256(trainfile.read_bytes()).hexdigest(),
        'existing_zero_lexical_support':sum(r['existing_zero'] for r in records),
        'top5_zero_lexical_support':sum(r['raw_top5_zero'] for r in records),
        'existing_lexical_edges':sum(r['existing_edges'] for r in records),
        'top5_lexical_edges':sum(r['raw_top5_edges'] for r in records),
        'missing_source_metadata_docs':sum('answers_found' not in d for r in raw.values() for d in r['docs'][:5]),
        'heldout_rows_analyzed':0,'model_generations':0,
        'supersedes_invalid_v1':'runs/qwen_training_retrieval_audit_20261010/summary.json',
        'invalid_v1_reason':'Source lacks answers_found; missing metadata was incorrectly counted as zero coverage.',
        'matching':'Same ALCE-normalized title+text alias substring as frozen prospective v2 data.',
        'warning':'Zero lexical edges are not proof that semantic evidence is absent. Source includes other rows but only the existing 350 train IDs are analyzed.',
        'per_row':records}
    out=ROOT/'runs/qwen_training_retrieval_audit_v2_20261010';out.mkdir(exist_ok=True)
    target=out/'summary.json'
    if target.exists():raise FileExistsError('Frozen audit exists')
    target.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)


if __name__=='__main__':main()

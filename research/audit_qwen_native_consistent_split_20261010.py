"""Exact ID/question split audit; no sealed inputs or model calls."""
import json,re,hashlib
from collections import defaultdict
from compile_qwen_native_consistent_20261010 import ROOT,RUN
paths={'train':RUN/'compiled/train.jsonl','regression_dev':ROOT/'data/qampari_evidence_full/dev.jsonl',
 'prospective_dev':ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'}
rows={k:[json.loads(s) for s in p.read_text().splitlines() if s.strip()] for k,p in paths.items()}
def key(r):return re.sub(r'\W+',' ',r['question'].lower()).strip()
maps={}
for name,v in rows.items():
 d=defaultdict(list)
 for r in v:d[key(r)].append(r['id'])
 maps[name]=d
comparisons={}
for a,b in [('train','regression_dev'),('train','prospective_dev'),('regression_dev','prospective_dev')]:
 common=set(maps[a])&set(maps[b]);ids=set(r['id'] for r in rows[a])&set(r['id'] for r in rows[b])
 comparisons[a+'_versus_'+b]={'id_overlap':sorted(ids),'normalized_question_overlap':[
  {'question_key':q,'a_ids':maps[a][q],'b_ids':maps[b][q]} for q in sorted(common)]}
out=RUN/'split_audit.json';assert not out.exists()
report={'status':'COMPLETE_EXACT_SPLIT_AUDIT','rows':{k:len(v) for k,v in rows.items()},
 'duplicate_questions_within_split':{k:{q:ids for q,ids in d.items() if len(ids)>1} for k,d in maps.items()},
 'comparisons':comparisons,'data_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
 'sealed_rows_read':0,'model_generations':0,'warning':'No approximate semantic/paraphrase duplicate certification; same source and document overlap may remain.'}
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

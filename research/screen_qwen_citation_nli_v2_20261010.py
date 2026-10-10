"""Fixed automatic citation screen; not official AutoAIS and not human evaluation."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib,json,time,re
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi,snapshot_download
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(str(_release_root));OUT=ROOT/'runs/qwen_citation_nli_screen_v2_20261010'
MODEL='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def main():
 if OUT.exists():raise FileExistsError('Refusing existing screen')
 OUT.mkdir();status=OUT/'status.txt';status.write_text('DOWNLOADING_JUDGE\n')
 revision='6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
 path=snapshot_download(MODEL,revision=revision,local_dir=str(_release_models / 'DeBERTa-v3-base-mnli-fever-anli'),
  allow_patterns=['config.json','model.safetensors','tokenizer*','spm.model','special_tokens_map.json','added_tokens.json','README.md'],max_workers=2)
 from verify_nli_snapshot_20261010 import verify
 provenance=verify(Path(path));(OUT/'judge_provenance.json').write_text(json.dumps(provenance,indent=2))
 tok=AutoTokenizer.from_pretrained(path);model=AutoModelForSequenceClassification.from_pretrained(path).to('cuda').eval()
 assert model.config.id2label[0].lower()=='entailment'
 cache={};cachefile=OUT/'judge_cache.jsonl';truncated=0
 def nli(premise,hypothesis):
  nonlocal truncated
  key=hashlib.sha256(json.dumps([premise,hypothesis],ensure_ascii=False).encode()).hexdigest()
  if key not in cache:
   full=tok(premise,hypothesis,truncation=False);cut=len(full['input_ids'])>512;truncated+=int(cut)
   inputs=tok(premise,hypothesis,truncation='only_first',max_length=512,return_tensors='pt').to('cuda')
   with torch.inference_mode():prob=model(**inputs).logits.float().softmax(-1)[0].cpu().tolist()
   v={'pair_sha256':key,'premise':premise,'hypothesis':hypothesis,'probabilities':prob,
      'entailment':int(np.argmax(prob)==0),'premise_truncated':cut}
   cache[key]=v
   with cachefile.open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
  return cache[key]['entailment']
 controls=[('Paris is the capital of France.','Paris is the capital of France.',1),
 ('Paris is the capital of France.','Berlin is the capital of France.',0),
 ('The film was directed by Alice.','The film was directed by Alice.',1),
 ('The film was directed by Alice, not Bob.','The film was directed by Bob.',0),
 ('Alice was born in London.','Where was Alice born? London.',1),
 ('Alice was born in London.','Where was Alice born? Berlin.',0)]
 control_results=[{'premise':p,'hypothesis':h,'expected':y,'observed':nli(p,h)} for p,h,y in controls]
 (OUT/'toy_controls.json').write_text(json.dumps(control_results,indent=2))
 assert all(v['expected']==v['observed'] for v in control_results),'Toy sanity failed: do not interpret screening scores'
 status.write_text('SCREENING_EXISTING_OUTPUTS\n')
 systems={'baseline':ROOT/'runs/submission_20261010/qwen3_frozen_comparison',
 'foundation':ROOT/'runs/qwen_retrieval_align_20261010/foundation',
 'clean_pair':ROOT/'runs/qwen_clean_labels_20261010/action_pair',
 'bridge_pair':ROOT/'runs/qwen_terminal_bridge_20261010/action_pair',
 **{a:ROOT/'runs/qwen_native_consistent_20261010'/a for a in ['action_pair','action_ce','action_ul']}}
 results={};inputhashes={}
 for arm,base in systems.items():
  if arm.startswith('action_'):
   status.write_text('WAITING_FOR_'+arm+'\n')
   for tick in range(480):
    state=(base/'status.txt').read_text().strip()
    if state=='FAILED':raise RuntimeError('Generation pipeline failed '+arm)
    if state=='COMPLETE':break
    time.sleep(15)
   else:raise TimeoutError('Generation incomplete')
  status.write_text('SCREENING_'+arm+'\n')
  results[arm]={}
  for setting in ['regression_dev','prospective_dev']:
   dp=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
   pp=base/setting/'eval_results.jsonl'
   if arm=='baseline':pp=base/setting/'baseline/eval_results.jsonl'
   data={r['id']:r for r in read(dp)};preds=read(pp);assert len(preds)==80 and {r['id'] for r in preds}==set(data)
   inputhashes[str(pp.relative_to(ROOT))]=hashlib.sha256(pp.read_bytes()).hexdigest()
   rows=[]
   for pred in preds:
    r=data[pred['id']];docs={d['id']:'Title: '+d['title']+'\n'+d['text'] for d in r['docs']}
    units=[u.strip() for u in pred['output'].rstrip().rstrip('.').rstrip(',').split(',') if re.sub(r'\[\d+\]','',u).strip()]
    admitted=supported=refs_total=missing=invalid=0;records=[]
    for u in units:
     refs=list(dict.fromkeys(int(x) for x in re.findall(r'\[(\d+)\]',u)));refs_total+=len(refs)
     claim=r['question']+' '+re.sub(r'\[\d+\]','',u).strip()
     valid=bool(refs) and all(i in docs for i in refs)
     if not refs:missing+=1
     if refs and not valid:invalid+=1
     joint=nli('\n'.join(docs[i] for i in refs),claim) if valid else 0;supported+=joint
     necessary=0
     if joint:
      for i in refs:
       solo=nli(docs[i],claim)
       without=nli('\n'.join(docs[j] for j in refs if j!=i),claim) if len(refs)>1 and not solo else 0
       necessary+=int(solo or not without)
     admitted+=necessary;records.append({'unit':u,'references':refs,'joint_supported':joint,'necessary_citations':necessary})
    rows.append({'id':r['id'],'units':len(units),'supported_units':supported,'references':refs_total,'necessary_references':admitted,
     'citation_support_rate':supported/max(1,len(units)),'citation_precision_screen':admitted/max(1,refs_total),
     'missing_citation_units':missing,'invalid_citation_units':invalid,'tokens':pred['tokens'],'unit_records':records})
   results[arm][setting]=rows
   (OUT/(arm+'_'+setting+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
   print(json.dumps({'arm':arm,'setting':setting,'n':80,'unique_nli_pairs':len(cache),
    'citation_support_rate':float(np.mean([r['citation_support_rate'] for r in rows])),
    'citation_precision_screen':float(np.mean([r['citation_precision_screen'] for r in rows]))}),flush=True)
 report={'status':'COMPLETE_AUTOMATIC_NLI_SCREEN','judge':MODEL,'revision':revision,'max_length':512,
  'decision':'entailment argmax, fixed before scoring; no tuned threshold','toy_controls_passed':6,
  'unique_nli_pairs':len(cache),'truncated_pairs':truncated,'predictions_screened':1120,'holdout_predictions':0,'human_annotations':0,
  'input_sha256':inputhashes,'table':{},'comparisons':{},'warnings':[
   'Exploratory DeBERTa screen, NOT official T5-XXL AutoAIS, not benchmark leaderboard.',
   'Question+answer claim and comma units follow ALCE-style convention; not validated decontextualization.',
   'Six toy controls are software sanity, not calibration or human agreement.',
   'NLI can confuse relation evidence, negation, names and lexical overlap; train-source overlap possible.',
   'Precision handles invalid refs conservatively in denominator; differs from official ALCE implementation.',
   'Judge only sees passage/claim; system names are never model input. Same reused development; question CI not seed CI.']}
 names=['citation_support_rate','citation_precision_screen']
 for setting in ['regression_dev','prospective_dev']:
  report['table'][setting]={a:{n:float(np.mean([r[n] for r in results[a][setting]])) for n in names} for a in systems}
  report['comparisons'][setting]={}
  for control in ['baseline','action_ce','action_ul','clean_pair','bridge_pair','foundation']:
   a={r['id']:r for r in results['action_pair'][setting]};b={r['id']:r for r in results[control][setting]};keys=sorted(a);assert a.keys()==b.keys()
   delta=np.array([[a[k][n]-b[k][n] for n in names] for k in keys]);rng=np.random.default_rng(20261010)
   boot=np.concatenate([delta[rng.integers(0,80,(1000,80))].mean(1) for _ in range(20)])
   report['comparisons'][setting][control]={n:{'delta':float(delta[:,i].mean()),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,n in enumerate(names)}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));status.write_text('COMPLETE\n')
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
 source=ROOT/'code/screen_qwen_citation_nli_20261010.py';hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
 (OUT/'completion_inventory.json').write_text(json.dumps(hashes,indent=2));members=list(hashes)+[str((OUT/'completion_inventory.json').relative_to(ROOT))]
 (ROOT/'exports/qwen_citation_nli_screen_v2_20261010_members.list').write_text('\n'.join(members)+'\n')
if __name__=='__main__':
 try:main()
 except Exception:
  if OUT.exists():(OUT/'status.txt').write_text('FAILED\n')
  raise

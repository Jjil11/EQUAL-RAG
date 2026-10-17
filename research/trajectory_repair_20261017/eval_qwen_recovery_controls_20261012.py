"""Frozen prompt-format controls and gated matched development comparisons."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse,hashlib,json,time
from types import SimpleNamespace
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from qwen_decision_recovery_20261012 import RUN,SOURCE,ROOT,MODEL,read
from eval_verified_outputs_20261010 import evaluate
from score_strict_units_20261010 import score,summarize

FORMAT='Each answer item must use square-bracket citations immediately after that item. Syntax example only (not answers to the question): Answer A [1], Answer B [2]. Do not put all citations in a separate list at the end.\n\n'
ANCHOR=ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'
ARMS=['full_sft','decision_kl','decision_no_kl']

def gates():
 result={}
 for arm in ARMS:
  p=RUN/arm/'status.txt'
  if not p.exists() or p.read_text().strip() not in {'COMPLETE','FAILED'}:return None
  assert p.read_text().strip()=='COMPLETE',f'{arm} failed; do not silently exclude'
  s=json.loads((RUN/arm/'train/diagnostic/summary.json').read_text())['tables']
  result[arm]=s['branch']['success']>=4 and s['prompt']['success']>=4
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=ARMS+['native_format','anchor_format']);args=ap.parse_args()
 arm=args.arm;out=RUN/'controls'/arm;assert not out.exists();out.mkdir(parents=True)
 status=out/'status.txt';status.write_text('WAITING\n')
 try:
  start=time.time();decision=None
  if arm in ARMS:
   while (decision:=gates()) is None:
    assert time.time()-start<7200,'Training wait timeout';time.sleep(20)
   if not any(decision.values()):
    (out/'gate_decision.json').write_text(json.dumps(decision,indent=2));status.write_text('SKIPPED_FAILED_TRAINING_GATE\n');return
  else:
   dependency='native_qwen' if arm=='native_format' else 'full_sft'
   while not (RUN/dependency/'status.txt').exists() or (RUN/dependency/'status.txt').read_text().strip()!='COMPLETE':
    assert time.time()-start<7200,'Dependency timeout';time.sleep(20)
  while torch.cuda.mem_get_info()[0]/2**30<32:
   assert time.time()-start<7200,'Free-memory timeout';time.sleep(20)
  status.write_text('LOADING\n')
  adapter=None if arm=='native_format' else (ANCHOR if arm=='anchor_format' else RUN/arm/'train/adapter')
  tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True);tok._equal_rag_disable_thinking=True
  model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16)
  if adapter is not None:model=PeftModel.from_pretrained(model,str(adapter))
  model=model.cuda().eval()
  settings=[('regression_dev',ROOT/'data/qampari_evidence_full/dev.jsonl'),('prospective_dev',ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl')]
  if arm.endswith('_format'):settings.insert(0,('training_diagnostic',SOURCE/'diagnostic16.jsonl'))
  manifest={'arm':arm,'model':MODEL,'adapter':str(adapter) if adapter else None,'format_prefix':FORMAT if arm.endswith('_format') else '',
   'max_new_tokens':256,'max_length':2048,'greedy':True,'disable_thinking':True,'gate_decision':decision,
   'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),'datasets':{},'sealed_predictions':0}
  evalargs=SimpleNamespace(disable_thinking=True,max_length=2048,max_new_tokens=256)
  for setting,data in settings:
   rows=read(data);assert len(rows)==(16 if setting=='training_diagnostic' else 80)
   manifest['datasets'][setting]={'source':str(data),'sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'n':len(rows)}
   if arm.endswith('_format'):rows=[{**r,'prompt':FORMAT+r['prompt']} for r in rows]
   status.write_text('EVALUATING_'+setting+'\n')
   target=out/setting;evaluate(evalargs,model,tok,rows,target)
   preds=read(target/'eval_results.jsonl');scored=[score(r,p) for r,p in zip(rows,preds)]
   assert [r['id'] for r in rows]==[p['id'] for p in preds]
   (target/'strict_metrics.json').write_text(json.dumps(summarize(scored),indent=2))
   (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
  status.write_text('COMPLETE\n')
 except Exception:
  status.write_text('FAILED\n');raise

if __name__=='__main__':main()

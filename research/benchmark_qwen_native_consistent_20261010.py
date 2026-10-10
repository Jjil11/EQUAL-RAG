"""Small same-GPU ABBA replay; generation latency, not serving throughput."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import gc,hashlib,json,time,platform
from pathlib import Path
import numpy as np
import torch,transformers,peft
from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(str(_release_root));RUN=ROOT/'runs/qwen_native_consistent_20261010'
OUT=ROOT/'runs/qwen_native_consistent_latency_20261010'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
@torch.inference_mode()
def main():
 assert not (OUT/'summary.json').exists()
 tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
 selected=[];reference={};shas={}
 for setting in ['regression_dev','prospective_dev']:
  dp=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
  rows=read(dp);chosen=sorted(rows,key=lambda r:hashlib.sha256(('latency-abba-20261010:'+r['id']).encode()).hexdigest())[:8]
  shas[setting]=hashlib.sha256(dp.read_bytes()).hexdigest()
  for r in chosen:r=dict(r);r['setting']=setting;selected.append(r)
  for arm in ['baseline','action_pair']:
   p=ROOT/'runs/submission_20261010/qwen3_frozen_comparison'/setting/'baseline' if arm=='baseline' else RUN/arm/setting
   reference[arm,setting]={r['id']:r['generated_token_ids'] for r in read(p/'termination_telemetry.jsonl')}
 (OUT/'protocol.json').write_text(json.dumps({'selected':[{k:r[k] for k in ['id','setting']} for r in selected],
  'order':['baseline','action_pair','action_pair','baseline'],'repeats_per_question_arm':2,'batch_size':1,'max_new_tokens':256,
  'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
  'python':platform.python_version(),'dtype':'bfloat16','input_sha256':shas,'selection':'8 per setting, SHA256 ID ordering, no output filtering'},indent=2))
 records=[]
 for block,arm in enumerate(['baseline','action_pair','action_pair','baseline']):
  assert torch.cuda.mem_get_info()[0]/2**30>32
  adapter=ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter' if arm=='baseline' else RUN/arm/'interp05/adapter'
  model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(_release_model),torch_dtype=torch.bfloat16),str(adapter)).cuda().eval()
  model.config.use_cache=True
  def encode(r):
   text=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   enc=tok(text,return_tensors='pt').to('cuda');assert enc['input_ids'].shape[-1]<=2048;return enc
  warm=encode(selected[0]);model.generate(**warm,max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id);torch.cuda.synchronize();del warm
  order=selected if block<2 else list(reversed(selected))
  for r in order:
   enc=encode(r);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   out=model.generate(**enc,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)
   torch.cuda.synchronize();elapsed=time.perf_counter()-start
   ids=out[0,enc['input_ids'].shape[-1]:].tolist()
   record={'id':r['id'],'setting':r['setting'],'arm':arm,'block':block,'seconds':elapsed,'tokens':len(ids),
    'input_tokens':enc['input_ids'].shape[-1],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'exact_reference_ids_match':ids==reference[arm,r['setting']][r['id']],'output_ids':ids}
   records.append(record)
   with (OUT/'replays.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print(json.dumps({k:record[k] for k in ['arm','block','id','seconds','tokens','exact_reference_ids_match']}),flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 report={'status':'COMPLETE_SMALL_SINGLE_GPU_ABBA_REPLAY','timed_generations':64,'warmup_generations':4,'unique_development_questions':16,
  'holdout_predictions':0,'all_exact_reference_ids_match':all(r['exact_reference_ids_match'] for r in records),'settings':{},
  'warnings':['Small reused development subset, two repetitions; not production throughput or reliable tail latency.',
   'Generation timer excludes CPU tokenization, model loading and retrieval; includes prefill+decode with CUDA synchronization.',
   'No exclusive whole-server reservation, other GPUs active; ABBA reduces but does not eliminate time/thermal effects.',
   'No training amortization or energy claim. Word/token reduction alone does not imply acceleration.']}
 for setting in ['regression_dev','prospective_dev']:
  keys=sorted({r['id'] for r in records if r['setting']==setting});a={}
  for arm in ['baseline','action_pair']:
   a[arm]=np.array([np.mean([r['seconds'] for r in records if r['arm']==arm and r['setting']==setting and r['id']==k]) for k in keys])
  rng=np.random.default_rng(20261010);ix=rng.integers(0,8,(20000,8));rel=a['action_pair'][ix].mean(1)/a['baseline'][ix].mean(1)-1
  report['settings'][setting]={'n':8,'mean_seconds':{arm:float(v.mean()) for arm,v in a.items()},
    'relative_latency_change':float(a['action_pair'].mean()/a['baseline'].mean()-1),'relative_ci95':np.quantile(rel,[.025,.975]).tolist()}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

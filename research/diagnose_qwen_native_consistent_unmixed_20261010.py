"""One training-only checkpoint diagnostic, no extra generation or tuning."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json,hashlib
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
ROOT=Path(str(_release_root));RUN=ROOT/'runs/qwen_native_consistent_20261010'
OUT=ROOT/'runs/qwen_native_consistent_unmixed_diagnostic_20261010'
@torch.inference_mode()
def main():
 assert torch.cuda.mem_get_info()[0]/2**30>32 and not (OUT/'summary.json').exists()
 src=RUN/'compiled/train.jsonl';rows=[json.loads(s) for s in src.read_text().splitlines() if s.strip()]
 rows=[r for r in rows if r.get('terminal_decision',{}).get('origin')=='recorded_onpolicy_prefix'];assert len(rows)==72
 model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(_release_model),torch_dtype=torch.bfloat16),str(RUN/'action_pair/train/adapter')).cuda().eval()
 values=[]
 for r in rows:
  d=r['terminal_decision'];inp=torch.tensor(d['context_ids'],device='cuda')[None]
  logits=model(input_ids=inp,use_cache=False).logits[0,-1].float();prob=logits.softmax(-1)
  values.append({'id':r['id'],'eos_argmax':int(logits.argmax()) in d['allowed_ids'],'eos_probability':float(prob[d['allowed_ids']].sum())})
  if len(values)%24==0:print('UNMIXED_DIAGNOSTIC',len(values),72,flush=True)
 report={'status':'COMPLETE_TRAIN_ONLY_UNMIXED_DIAGNOSTIC','n':72,'checkpoint':'action_pair/train/adapter before fixed interpolation',
  'eos_argmax_count':sum(r['eos_argmax'] for r in values),'mean_eos_probability':sum(r['eos_probability'] for r in values)/72,
  'compiled_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'model_generations':0,'heldout_or_development_rows_read':0,
  'warning':'Diagnostic only: not selecting interpolation or evaluating another variant on development; do not promote to method result.',
  'rows':values}
 (OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':main()

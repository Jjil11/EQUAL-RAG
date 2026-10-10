"""Training-only invariance and exact action-path checks, no model generation."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib,json
from transformers import AutoTokenizer
from compile_qwen_native_consistent_20261010 import ROOT,RUN,INVALID
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
old={r['id']:r for r in read(INVALID/'compiled/train.jsonl')};rows=read(RUN/'compiled/train.jsonl')
assert len(old)==len(rows)==350 and {r['id'] for r in rows}==old.keys()
tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
fields=['prompt','short','edges','cache_metadata','failure_local_retention_mask','completion_pair','generation_token_ids']
native=active=0;flags=[]
for r in rows:
 assert all(r.get(k)==old[r['id']].get(k) for k in fields)
 if not r['has_trajectory']:continue
 active+=1;d=r['terminal_decision'];a=r['action_decision']
 assert set(d['allowed_ids'])=={151643,151645} and d['canonical_eos_id'] in d['allowed_ids']
 assert a['allowed_ids'] and a['rejected_id'] not in a['allowed_ids'] and len(a['context_ids'])<2048 and len(d['context_ids'])<2048
 positive=d['context_ids']+[d['canonical_eos_id']];c=a['context_ids'];bad=a['rejected_id']
 assert not (positive[:len(c)]==c and len(positive)>len(c) and positive[len(c)]==bad)
 if d['origin']=='recorded_onpolicy_prefix':
  native+=1;k=d['generation_prefix_tokens'];prompt=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  assert a['context_ids']==d['context_ids']==prompt+r['generation_token_ids'][:k]
  assert a['allowed_ids']==d['allowed_ids'] and bad==r['generation_token_ids'][k]
 else:
  assert a==old[r['id']]['action_decision']
  if tok.decode(d['context_ids'],skip_special_tokens=True).startswith(tok.decode(c+[bad],skip_special_tokens=True)):
   flags.append({'id':r['id'],'first_context_tail':tok.decode(c[-60:]),'bad_token':tok.decode([bad]),
     'allowed_tokens':[tok.decode([i]) for i in a['allowed_ids']],
     'terminal_context_tail':tok.decode(d['context_ids'][-80:]),'exact_token_path_conflict':False})
assert active==150 and native==72 and len(flags)==4
out=RUN/'training_consistency_audit.json';assert not out.exists()
v={'status':'PASS_EXACT_TOKEN_AND_INVARIANCE_CHECKS','rows':350,'active':150,'native_stop':72,
 'unchanged_fields':fields,'exact_token_path_conflicts':0,'decoded_only_flags':flags,
 'development_or_holdout_rows_read':0,'compiled_sha256':hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest(),
 'warning':'Four unchanged counterfactuals have decoded-prefix flags: alternate tokenization, not an exact token conflict; no semantic validity guarantee.'}
out.write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2),flush=True)

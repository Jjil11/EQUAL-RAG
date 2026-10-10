"""Align first stop action and terminal EOS on actual on-policy prefixes."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from compile_qwen_terminal_bridge_20261010 import ROOT,CACHE,RUN as SOURCE
RUN=ROOT/'runs/qwen_native_consistent_20261010'
INVALID=ROOT/'runs/qwen_native_terminal_20261010'
def main():
 if RUN.exists():raise FileExistsError('Refusing overwrite')
 src=INVALID/'compiled/train.jsonl';sha=hashlib.sha256(src.read_bytes()).hexdigest()
 assert sha=='2b7a0afdc75f2c2c0e1af3b967774512e020ebdca51712a5ab261de609c692ff'
 rows=[json.loads(s) for s in src.read_text().splitlines() if s.strip()]
 tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
 changed=[];decoded=[];exact=[]
 for r in rows:
  if not r['has_trajectory']:continue
  d=r['terminal_decision'];a=r['action_decision']
  if d['origin']=='recorded_onpolicy_prefix':
   k=d['generation_prefix_tokens'];bad=r['generation_token_ids'][k]
   assert r['completion_pair']['action']=='stop' and bad not in d['allowed_ids']
   r['previous_inconsistent_action_decision']=dict(a)
   a.update(context_ids=d['context_ids'],allowed_ids=d['allowed_ids'],rejected_id=bad,
            origin='native_eos_vs_actual_next_repeat_token')
   changed.append(r['id'])
  positive=d['context_ids']+[d['canonical_eos_id']];c=a['context_ids'];bad=a['rejected_id']
  if positive[:len(c)]==c and len(positive)>len(c) and positive[len(c)]==bad:exact.append(r['id'])
  if tok.decode(d['context_ids'],skip_special_tokens=True).startswith(tok.decode(c+[bad],skip_special_tokens=True)):decoded.append(r['id'])
 assert len(rows)==350 and len(changed)==72 and not exact
 assert sum(r['has_trajectory'] for r in rows)==150 and len(decoded)==4
 out=RUN/'compiled';out.mkdir(parents=True);p=out/'train.jsonl'
 p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 active=[r for r in rows if r['has_trajectory']]
 smoke=[r for r in active if r['terminal_decision']['origin']=='recorded_onpolicy_prefix'][:4]+[r for r in active if r['completion_pair']['action']=='complete_then_stop'][:4]
 (out/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke))
 audit={'rows':350,'active':150,'explicit_eos_sites':150,'native_eos_sites':72,'counterfactual_eos_sites':78,
 'stop':75,'completion':75,'zero_lexical_support':106,'smoke_rows':8,'source_sha256':sha,
 'compiled_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changed_ids':changed,
 'exact_token_path_conflicts':exact,'decoded_only_prefix_flags':decoded,'development_or_holdout_inputs_read':0,
 'warning':'Joint first-action/terminal consistency repair, not isolated EOS representation ablation. Four unchanged counterfactual labels have decoded-prefix flags under different tokenization; no exact rejected token on positive path.'}
 (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)
if __name__=='__main__':main()

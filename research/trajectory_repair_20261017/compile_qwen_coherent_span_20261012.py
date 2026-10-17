"""Train-only contiguous repair suffix from recorded model-token divergence."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer,GenerationConfig
from completion_pairs_20261010 import ROOT,encode_end,covered,true_repeat

SOURCE=ROOT/'runs/qwen_native_consistent_20261010'
RUN=ROOT/'runs/qwen_coherent_span_20261012'
CACHE=Path(str(_release_cache / 'equal_rag_cache/qwen_retrieval_align_20261010'))
SOURCE_SHA='cbcf5dce1fbfa74bb75142fb9460c784d60ee891e231d6ad03b5b4e520c1d156'

def common_prefix(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n

def main():
 assert not RUN.exists(),'Refusing overwrite'
 src=SOURCE/'compiled/train.jsonl'
 assert hashlib.sha256(src.read_bytes()).hexdigest()==SOURCE_SHA
 rows=[json.loads(s) for s in src.read_text().splitlines() if s.strip()]
 tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
 tok._equal_rag_disable_thinking=True
 stops=GenerationConfig.from_pretrained(tok.name_or_path).eos_token_id
 stops=[stops] if isinstance(stops,int) else list(stops)
 diagnostics=[]
 for r in rows:
  if not r['has_trajectory']:continue
  chosen=r['completion_pair']['chosen']
  assert len(covered(chosen,r['edges']))==len(r['edges']) and not true_repeat({'short':chosen})
  positive=encode_end(tok,r['prompt'],chosen,stops)[0].tolist()
  prompt=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  actual=prompt+r['generation_token_ids']
  assert positive[:len(prompt)]==prompt,'Chat-template prefix mismatch'
  n=common_prefix(positive,actual)
  assert len(prompt)<=n<min(len(positive),len(actual)),r['id']
  assert len(positive)<=2048 and positive[-1] in stops and not set(stops).intersection(positive[len(prompt):-1])
  # Never put a rejected next token on the positive token path.
  rejected=actual[n];allowed=stops if positive[n] in stops else [positive[n]]
  assert rejected not in allowed
  context=actual[:n];assert context==positive[:n]
  span={'positive_ids':positive,'shared_tokens':n,'prompt_tokens':len(prompt),
        'allowed_first_ids':allowed,'rejected_first_id':rejected,'eos_ids':stops,
        'suffix_tokens':len(positive)-n,'body_tokens':len(positive)-n-1,
        'origin':'exact_lcp_of_recorded_generation_and_verified_counterfactual'}
  r['coherent_span']=span
  diagnostics.append({'id':r['id'],'action':r['completion_pair']['action'],
    'native_stop':r['terminal_decision']['origin']=='recorded_onpolicy_prefix',
    'shared_generation_tokens':n-len(prompt),'suffix_tokens':span['suffix_tokens'],
    'body_tokens':span['body_tokens'],'first_positive_id':positive[n],'first_rejected_id':rejected,
    'first_positive_text':tok.decode([positive[n]]),'first_rejected_text':tok.decode([rejected])})
 assert len(rows)==350 and len(diagnostics)==150
 out=RUN/'compiled';out.mkdir(parents=True)
 train=out/'train.jsonl';train.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 native=[r for r in rows if r['has_trajectory'] and r['terminal_decision']['origin']=='recorded_onpolicy_prefix']
 complete=[r for r in rows if r['has_trajectory'] and r['completion_pair']['action']=='complete_then_stop']
 smoke=native[:4]+complete[:4]
 (out/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke))
 report={'status':'COMPLETE_TRAIN_ONLY_COMPILE','rows':350,'active':150,'native_stop':len(native),
  'completion_pairs':len(complete),'smoke_rows':8,'exact_path_conflicts':0,
  'source_sha256':SOURCE_SHA,'compiled_sha256':hashlib.sha256(train.read_bytes()).hexdigest(),
  'model_generations':0,'development_or_holdout_rows_read':0,
  'mean_suffix_tokens':sum(v['suffix_tokens'] for v in diagnostics)/150,
  'max_suffix_tokens':max(v['suffix_tokens'] for v in diagnostics),
  'suffix_one_token':sum(v['suffix_tokens']==1 for v in diagnostics),
  'unchanged_fields':['prompt','short','edges','generation_token_ids','failure_local_retention_mask','cache_metadata'],
  'warnings':['Training labels remain lexical proxies, not semantic certificates.',
    'Joint change of repair context and contiguous suffix; not a single-factor ablation of prior sparse version.',
    'No new retrieval inputs or teacher cache, no changed learning rate/epochs/interpolation.'],
  'per_row':diagnostics}
 (out/'audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)

if __name__=='__main__':main()

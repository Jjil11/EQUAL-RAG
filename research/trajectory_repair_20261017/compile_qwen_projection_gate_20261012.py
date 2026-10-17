"""Train-only projection/grammar feasibility gate; never reads dev or sealed data."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib,json
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer,GenerationConfig
from completion_pairs_20261010 import ROOT,norm,units,covered,true_repeat,edge_matches,encode_end
from compile_qwen_coherent_span_20261012 import common_prefix

RUN=ROOT/'runs/qwen_projection_gate_v2_20261012'
SOURCE=ROOT/'runs/qwen_coherent_span_20261012/compiled/train.jsonl'
CANDIDATES=ROOT/'runs/qwen_verified_projection_preparation_20261012/positive_candidates.jsonl'
SOURCE_SHA='7beb863bf33f9043a75440cc56f0c963d85474d8d475494e4301a3b7e91a9cb9'
CANDIDATE_SHA='dccea52f2552e38f643935a17fef8598eec12781fbcfc209578c94118d4a4710'

def main():
 assert not RUN.exists(),'Refusing overwrite'
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
 assert hashlib.sha256(CANDIDATES.read_bytes()).hexdigest()==CANDIDATE_SHA
 rows=[json.loads(s) for s in SOURCE.read_text().splitlines() if s.strip()]
 targets={r['id']:r for r in map(json.loads,CANDIDATES.read_text().splitlines())}
 tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
 tok._equal_rag_disable_thinking=True
 stops=GenerationConfig.from_pretrained(tok.name_or_path).eos_token_id
 stops=[stops] if isinstance(stops,int) else list(stops)
 collision_groups=guard_rows=contrast_rows=0
 for r in rows:
  if not r['has_trajectory']:continue
  grouped=defaultdict(list)
  for e in r['edges']:grouped[norm(e['answer'])].append(e)
  for es in grouped.values():
   if len(es)>1:
    collision_groups+=1
    assert len({tuple(sorted({norm(a) for a in e['aliases'] if norm(a)})) for e in es})==1,'Ambiguous canonical merge'
  prompt=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  actual=prompt+r['generation_token_ids'];seqs=[];texts=[]
  for c in targets[r['id']]['canonical_candidates']:
   text=c['text'];p=encode_end(tok,r['prompt'],text,stops)[0].tolist()
   assert p[:len(prompt)]==prompt and len(p)<=2048
   assert len(covered(text,r['edges']))==len(r['edges']) and not true_repeat({'short':text})
   assert all(edge_matches(u,r['edges']) for _,_,u in units(text))
   seqs.append(p);texts.append(text)
  n=min(common_prefix(p,actual) for p in seqs)
  assert len(prompt)<=n<min(map(len,seqs)) and n<len(actual)
  allowed=sorted(set(p[n] for p in seqs));rejected=actual[n]
  use_contrast=rejected not in allowed
  contrast_rows+=use_contrast
  # A verified first citation unit is a factual prefix, not a complete answer
  # when at least one known edge remains uncovered. Discourage EOS there.
  guard=None;first=units(texts[0])[0]
  prefix=texts[0][:first[1]]
  if len(covered(prefix,r['edges']))<len(r['edges']):
   # Slice the ORIGINAL full path, not a retokenized closed prefix: BPE can
   # merge a citation bracket with a comma. Include that boundary token.
   _,offsets,start=encode_end(tok,r['prompt'],texts[0],stops)
   shared=next(i+1 for i,(lo,hi) in enumerate(offsets) if hi>=start+first[1] and hi>lo)
   ids=seqs[0][:shared]
   visible=tok.decode(ids[len(prompt):],skip_special_tokens=True)
   assert len(covered(visible,r['edges']))<len(r['edges'])
   assert all(p[:shared]==ids for p in seqs) and all(p[shared] not in stops for p in seqs)
   guard={'context_ids':ids,'allowed_next_ids':sorted({p[shared] for p in seqs}),'rejected_eos_ids':stops,
          'covered_edges':len(covered(visible,r['edges'])),'total_edges':len(r['edges']),
          'origin':'original_positive_token_path_after_first_closed_citation_including_merged_boundary'}
   guard_rows+=1
  r['verified_projection']={'positive_ids':seqs,'positive_texts':texts,'shared_tokens':n,
   'prompt_tokens':len(prompt),'allowed_first_ids':allowed,'rejected_first_id':rejected,
   'contrast_eligible':use_contrast,'eos_ids':stops,'incomplete_guard':guard}
 active=[r for r in rows if r['has_trajectory']]
 key=lambda r:hashlib.sha256(('projection-gate-20261012:'+r['id']).encode()).hexdigest()
 native=sorted([r for r in active if r['terminal_decision']['origin']=='recorded_onpolicy_prefix'],key=key)[:16]
 complete=sorted([r for r in active if r['completion_pair']['action']=='complete_then_stop'],key=key)[:16]
 selected=native+complete;assert len(selected)==len({r['id'] for r in selected})==32
 diag=native[:8]+complete[:8]
 out=RUN/'compiled';out.mkdir(parents=True)
 for name,values in [('train350',rows),('pilot32',selected),('diagnostic16',diag)]:
  (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
 report={'status':'COMPLETE_TRAIN_ONLY_COMPILATION','source_sha256':SOURCE_SHA,'candidates_sha256':CANDIDATE_SHA,
  'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')},
  'rows':350,'active':150,'pilot_rows':32,'diagnostic_rows':16,'canonical_collision_groups':collision_groups,
  'canonical_collision_alias_sets_identical':True,'contrast_eligible':contrast_rows,'incomplete_guard_rows':guard_rows,
  'development_or_holdout_rows_read':0,'model_generations':0,'epochs_pilot':3,'epochs_full_if_gate_passes':1,
  'gate':{'success_definition':'all known lexical edges + every emitted citation unit verified + no repeat + true EOS',
   'minimum_branch_success_pair':4,'minimum_branch_improvement_vs_anchor':4,
   'minimum_prompt_success_pair':4,'minimum_prompt_improvement_vs_anchor':2,
   'decision':'All four thresholds required before full350 development experiment; diagnostics are training-only, NOT final quality evidence.'},
  'warnings':['Lexical graph feasibility is not a semantic or entity-identity certificate.',
   'Two equal-weight grammar candidates, .1 body CE + .1 EOS CE, contrast/early-EOS margins .1 each.',
   'KL retention masks and teacher caches kept unchanged to isolate target/constraint pilot; their reliability remains a gap.',
   'Joint projection/grammar/early-stop-target change vs previous version, not a single-factor ablation.',
   'Pilot32 overfit feasibility only; any full run restarts from anchor, never pilot adapter.']}
 (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

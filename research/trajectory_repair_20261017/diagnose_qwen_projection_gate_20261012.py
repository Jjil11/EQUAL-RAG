"""Training-only free rollouts from branch and original prompt; no quality claim."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse,hashlib,json,re,time
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from completion_pairs_20261010 import ROOT,covered,true_repeat,units,norm,edge_matches
from compile_qwen_projection_gate_20261012 import RUN
from score_strict_units_20261010 import score,summarize

def outcome(row,text,ids,stops,cap):
 us=units(text);all_refs=True
 for _,_,u in us:
  refs={int(x) for x in re.findall(r'\[(\d+)\]',u)}
  matched=edge_matches(u,row['edges']);supported=set()
  for i in matched:supported.update(row['edges'][i].get('supporting_doc_ids',[row['edges'][i]['doc_id']]))
  all_refs=all_refs and bool(matched) and refs<=supported
 tail=text[us[-1][1]:] if us else text
 tail_clean=not re.sub(r'[\s.,;]', '',tail)
 all_covered=len(covered(text,row['edges']))==len(row['edges'])
 eos=bool(ids and ids[-1] in stops);repeated=bool(true_repeat({'short':text}))
 return {'tokens':len(ids),'ended_by_eos':eos,'true_cap_hit':len(ids)>=cap and not eos,
  'all_known_edges_covered':all_covered,'every_citation_unit_verified':all_refs and bool(us),
  'no_dangling_text':tail_clean,'repeated':repeated,
  'success':all_covered and all_refs and bool(us) and tail_clean and not repeated and eos}

@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True,choices=['baseline','projection_pair','projection_ce']);ap.add_argument('--adapter',required=True)
 args=ap.parse_args();out=RUN/'diagnostics'/args.arm;assert not out.exists();out.mkdir(parents=True)
 audit=json.loads((RUN/'compiled/audit.json').read_text());src=RUN/'compiled/diagnostic16.jsonl'
 assert hashlib.sha256(src.read_bytes()).hexdigest()==audit['hashes'][src.name]
 rows=[json.loads(s) for s in src.read_text().splitlines()];assert len(rows)==16
 modelpath=str(_release_model)
 tok=AutoTokenizer.from_pretrained(modelpath,use_fast=True);tok._equal_rag_disable_thinking=True
 stops=GenerationConfig.from_pretrained(modelpath).eos_token_id;stops=[stops] if isinstance(stops,int) else list(stops)
 assert torch.cuda.mem_get_info()[0]/2**30>32
 model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16),args.adapter).cuda().eval()
 values=[];started=time.time()
 for row in rows:
  d=row['verified_projection'];n=d['shared_tokens'];paths=d['positive_ids'];r={'id':row['id'],'action':row['completion_pair']['action']}
  contexts={'first':(paths[0][:n],d['allowed_first_ids']),
            'plain_terminal':(paths[0][:-1],stops),'period_terminal':(paths[1][:-1],stops)}
  g=d['incomplete_guard']
  if g:contexts['incomplete_eos']=(g['context_ids'],stops)
  for name,(context,allowed) in contexts.items():
   logits=model(input_ids=torch.tensor(context,device='cuda')[None],use_cache=False).logits[0,-1].float()
   r[name]={'allowed_probability':float(logits.softmax(-1)[allowed].sum()),'argmax_legal':int(logits.argmax()) in allowed}
   del logits
  for mode,context,cap in [('branch',paths[0][:n],64),('prompt',paths[0][:d['prompt_tokens']],256)]:
   generated=model.generate(input_ids=torch.tensor(context,device='cuda')[None],max_new_tokens=cap,do_sample=False,pad_token_id=tok.eos_token_id)
   ids=generated[0,len(context):].tolist();text=tok.decode(context[d['prompt_tokens']:]+ids,skip_special_tokens=True).strip()
   r[mode]={'output':text,'generated_token_ids':ids,**outcome(row,text,ids,stops,cap)};del generated
  values.append(r)
  with (out/'rollouts.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  print(json.dumps({'arm':args.arm,'done':len(values),'of':16,'branch_success':sum(v['branch']['success'] for v in values),'prompt_success':sum(v['prompt']['success'] for v in values)}),flush=True)
 table={}
 for mode in ['branch','prompt']:
  preds=[score(row,{'output':v[mode]['output'],'tokens':v[mode]['tokens'],'grounded_edges':0}) for row,v in zip(rows,values)]
  table[mode]={'n':16,'success':sum(v[mode]['success'] for v in values),
   'true_cap_hit_rate':float(np.mean([v[mode]['true_cap_hit'] for v in values])),
   'all_known_edges_covered':sum(v[mode]['all_known_edges_covered'] for v in values),
   'verified_units_only':sum(v[mode]['every_citation_unit_verified'] and v[mode]['no_dangling_text'] for v in values),
   **summarize(preds)}
 report={'status':'COMPLETE_TRAINING_ONLY_DIAGNOSTIC','arm':args.arm,'table':table,
  'first_argmax_legal':sum(v['first']['argmax_legal'] for v in values),
  'plain_terminal_eos_argmax':sum(v['plain_terminal']['argmax_legal'] for v in values),
  'period_terminal_eos_argmax':sum(v['period_terminal']['argmax_legal'] for v in values),
  'mean_first_allowed_probability':float(np.mean([v['first']['allowed_probability'] for v in values])),
  'training_model_generations':32,'development_or_holdout_generations':0,'human_annotations':0,
  'elapsed_seconds':time.time()-started,'diagnostic_sha256':audit['hashes'][src.name],
  'warnings':['Training cases, not quality confirmation or confidence intervals.',
   'Branch token cost counts only new continuation; do not compare it to full-prompt token cost.',
   'Lexical support and comma-list F1 do not certify relation entailment or split names containing commas correctly.']}
 (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

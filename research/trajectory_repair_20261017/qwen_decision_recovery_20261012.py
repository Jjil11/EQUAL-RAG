"""Matched training feasibility: full-response SFT, decision CE, anchoring ablation."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import train_trajectory_equal as legacy
from completion_pairs_20261010 import ROOT
from diagnose_qwen_projection_gate_20261012 import outcome
from score_strict_units_20261010 import score,summarize

RUN=ROOT/'runs/qwen_decision_recovery_20261012'
SOURCE=ROOT/'runs/qwen_projection_gate_v2_20261012/compiled'
MODEL=str(_release_model)
ARMS={'full_sft','decision_kl','decision_no_kl'}

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

@torch.inference_mode()
def diagnose(model,tok,rows,out):
 assert not out.exists();out.mkdir(parents=True)
 model.eval();model.config.use_cache=True
 if hasattr(model,'gradient_checkpointing_disable'):model.gradient_checkpointing_disable()
 device=next(model.parameters()).device;values=[]
 for row in rows:
  d=row['verified_projection'];paths=d['positive_ids'];n=d['shared_tokens'];stops=d['eos_ids'];v={'id':row['id']}
  for name,(context,allowed) in {'first':(paths[0][:n],d['allowed_first_ids']),
        'plain_terminal':(paths[0][:-1],stops),'period_terminal':(paths[1][:-1],stops)}.items():
   logits=model(input_ids=torch.tensor(context,device=device)[None],use_cache=False).logits[0,-1].float()
   v[name]={'argmax_legal':int(logits.argmax()) in allowed,'probability':float(logits.softmax(-1)[allowed].sum())};del logits
  for mode,context,cap in [('branch',paths[0][:n],64),('prompt',paths[0][:d['prompt_tokens']],256)]:
   start=time.time();generated=model.generate(input_ids=torch.tensor(context,device=device)[None],max_new_tokens=cap,do_sample=False,pad_token_id=tok.eos_token_id)
   ids=generated[0,len(context):].tolist();text=tok.decode(context[d['prompt_tokens']:]+ids,skip_special_tokens=True).strip()
   v[mode]={'output':text,'generated_token_ids':ids,**outcome(row,text,ids,stops,cap),'elapsed_seconds':time.time()-start};del generated
  values.append(v)
  with (out/'rollouts.jsonl').open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n')
  print(json.dumps({'phase':'diagnostic','done':len(values),'n':len(rows),
   'branch_success':sum(r['branch']['success'] for r in values),'prompt_success':sum(r['prompt']['success'] for r in values)}),flush=True)
 tables={}
 for mode in ['branch','prompt']:
  scored=[score(r,{'output':v[mode]['output'],'tokens':v[mode]['tokens'],'grounded_edges':0}) for r,v in zip(rows,values)]
  tables[mode]={**summarize(scored),'success':sum(v[mode]['success'] for v in values),
   'true_cap_hit_rate':float(np.mean([v[mode]['true_cap_hit'] for v in values])),
   'missing_known_edges':sum(not v[mode]['all_known_edges_covered'] for v in values),
   'unverified_units_or_refs':sum(not v[mode]['every_citation_unit_verified'] for v in values)}
 summary={'status':'COMPLETE_TRAINING_DIAGNOSTIC','n':len(rows),'model_generations':len(rows)*2,'tables':tables,
  'first_argmax_legal':sum(v['first']['argmax_legal'] for v in values),
  'plain_terminal_eos_argmax':sum(v['plain_terminal']['argmax_legal'] for v in values),
  'period_terminal_eos_argmax':sum(v['period_terminal']['argmax_legal'] for v in values),
  'warnings':['Training subset chosen before this round; no claim of unseen-data performance.',
   'Prepared grammar actions are not exhaustive valid expressions; lexical support is not semantic entailment.']}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
 return summary

def train(args,model,tokenizer,rows,anchor_model=None):
 arm=os.environ['EQUAL_DECISION_ARM'];assert arm in ARMS
 assert args.seed==20260930 and args.lr==1e-4 and args.epochs==6 and len(rows)==32
 assert args.grad_accum==8 and args.max_length==2048
 assert hashlib.sha256(Path(args.train).read_bytes()).hexdigest()=='de00ad3035b4ac7315701a7f4502678b9a5c70c3fca9cd82dc9fc0044d2655ae'
 loader=DataLoader(legacy.Rows(rows),batch_size=1,shuffle=True,collate_fn=legacy.one)
 opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
 model.train();model.config.use_cache=False;opt.zero_grad(set_to_none=True)
 device=next(model.parameters()).device;count=0;history=[];totals={'ce':0.,'decision_ce':0.,'kl':0.};started=time.time()
 for epoch in range(args.epochs):
  for row in loader:
   d=row['verified_projection'];prompt=d['prompt_tokens'];n=d['shared_tokens'];cevalue=decisionvalue=klvalue=0.
   mask=torch.tensor(row['failure_local_retention_mask'],device=device,dtype=torch.bool)
   if mask.any():
    _,_,_,student=legacy.response_logp(model,tokenizer,row['prompt'],row['short'],args.max_length,True)
    anchor=torch.load(legacy.anchor_cache_path(args.anchor_logits_dir,row['id']),map_location=device,weights_only=True)
    assert student.shape==anchor.shape and len(mask)==len(student)
    a=F.log_softmax(anchor[mask].float(),-1);s=F.log_softmax(student[mask].float(),-1)
    kl=(a.exp()*(a-s)).sum(-1).mean();klvalue=float(kl.detach())
    penalty=10*F.relu(kl-.0001) if arm!='decision_no_kl' else kl*0
    (penalty/args.grad_accum).backward();del student,anchor,a,s,kl,penalty
   for path in d['positive_ids']:
    ids=torch.tensor(path,device=device)[None];lp=F.log_softmax(model(input_ids=ids).logits[0,prompt-1:-1].float(),-1)
    targets=ids[0,prompt:];token_ce=-lp.gather(-1,targets[:,None]).squeeze(-1)
    # True assistant EOS mass; no post-EOS newline or prompt supervision.
    terminal=-torch.logsumexp(lp[-1,d['eos_ids']],0)
    ce=torch.cat([token_ce[:-1],terminal[None]]).mean()
    decisions=[-torch.logsumexp(lp[n-prompt,d['allowed_first_ids']],0),terminal]
    guard=d['incomplete_guard']
    if guard:
     at=len(guard['context_ids'])-prompt
     assert path[:len(guard['context_ids'])]==guard['context_ids']
     decisions.append(-torch.logsumexp(lp[at,guard['allowed_next_ids']],0))
    decision=torch.stack(decisions).mean()
    loss=(ce+(decision if arm!='full_sft' else decision*0))/len(d['positive_ids'])
    cevalue+=float(ce.detach())/2;decisionvalue+=float(decision.detach())/2
    (loss/args.grad_accum).backward();del ids,lp,targets,token_ce,terminal,ce,decisions,decision,loss
   count+=1
   totals['ce']+=cevalue;totals['decision_ce']+=decisionvalue;totals['kl']+=klvalue
   if count%args.grad_accum==0:
    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(grad)
    opt.step();opt.zero_grad(set_to_none=True)
    rec={'arm':arm,'epoch':epoch+1,'sample_visits':count,'total_visits':192,'optimizer_step':count//8,
     'ce_mean_so_far':totals['ce']/count,'decision_ce_mean_so_far':totals['decision_ce']/count,
     'kl_mean_so_far':totals['kl']/count,'gradient_norm':grad,'elapsed_seconds':time.time()-started,
     'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
    history.append(rec);print(json.dumps(rec),flush=True)
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 (out/'recovery_training_audit.json').write_text(json.dumps({'arm':arm,'unique_rows':32,'sample_visits':count,
  'epochs':6,'optimizer_steps':24,'lr':1e-4,'full_response_ce':True,'decision_ce_weight':0 if arm=='full_sft' else 1,
  'kl_weight':0 if arm=='decision_no_kl' else 10,'interpolation_weight':1.0,
  'same_positive_grammar_paths':True,'same_forward_calls':True,'gradient_norms_or_walltime_matched':False,
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'warnings':['Joint budget/full-answer/loss-weight change vs prior pilot, not isolated superiority over it.',
   'Current arms match training budget and positive labels; no-KL only changes KL gradient.']},indent=2))
 return history

def evaluate(args,model,tokenizer,rows,out_dir):
 assert len(rows)==16
 return diagnose(model,tokenizer,rows,Path(out_dir)/'diagnostic')

if __name__=='__main__':
 legacy.train=train;legacy.evaluate=evaluate;legacy.main()

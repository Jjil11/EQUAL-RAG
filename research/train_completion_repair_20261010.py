"""Matched same-label SFT vs completion-pair training, with selective KL.

Pilot only: no novelty or universal quality guarantee is asserted by this code.
"""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import train_trajectory_equal as legacy
from train_verified_boundary_20261010 import eos_ids
from completion_pairs_20261010 import compiled_pair

ARM=os.environ.get('EQUAL_COMPLETION_ARM','pair')
assert ARM in {'pair','sft'}


def branch_score(model,ids,n,stops):
    device=next(model.parameters()).device
    logits=model(input_ids=ids[None].to(device)).logits[0,n-1:-1].float()
    lp=F.log_softmax(logits,-1)
    targets=ids[n:].to(device)
    values=lp.gather(-1,targets[:,None]).squeeze(-1)
    # Both arms use the same legal EOS class. No post-EOS newline credit.
    values=torch.cat([values[:-1],torch.logsumexp(lp[-1,stops],0)[None]])
    return values.mean()


def train(args,model,tokenizer,rows,anchor_model=None):
    assert args.method=='trajectory_repair' and args.pref_alpha==args.stop_alpha==args.dual_max==0
    loader=DataLoader(legacy.Rows(rows),batch_size=1,shuffle=True,collate_fn=legacy.one)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    model.train(); model.config.use_cache=False;opt.zero_grad(set_to_none=True)
    stops=eos_ids(tokenizer);history=[];active=0;count=0;last_grad=None
    action_counts={'stop':0,'complete_then_stop':0}
    for epoch in range(args.epochs):
        for row in loader:
            mask=torch.tensor(row['retention_mask'],device=next(model.parameters()).device,dtype=torch.bool)
            kl_value=0.;loss_value=0.
            if mask.any():
                _,_,_,student=legacy.response_logp(model,tokenizer,row['prompt'],row['short'],args.max_length,True)
                anchor=torch.load(legacy.anchor_cache_path(args.anchor_logits_dir,row['id']),map_location=student.device,weights_only=True)
                if student.shape!=anchor.shape or len(mask)!=len(student):raise ValueError('Cache/mask alignment error')
                a=F.log_softmax(anchor[mask].float(),-1);s=F.log_softmax(student[mask].float(),-1)
                kl=(a.exp()*(a-s)).sum(-1).mean()
                safety=args.anchor_alpha*F.relu(kl-args.anchor_kl_target)
                kl_value=float(kl.detach());loss_value=float(safety.detach())
                (safety/args.grad_accum).backward()
                del student,anchor,a,s,kl,safety
            record={'id':row['id'],'arm':ARM,'anchor_kl':kl_value,'retained_tokens':int(mask.sum())}
            if row.get('has_trajectory'):
                chosen,rejected,n,spec=compiled_pair(tokenizer,row,stops,args.max_length)
                if spec!=row['completion_pair']:raise ValueError('Compiled branch changed')
                active+=1;action_counts[spec['action']]+=1
                cp=branch_score(model,chosen,n,stops)
                # Additive same-target CE in both arms. The pair arm adds only
                # the explicit counterfactual preference; teacher is detached.
                pair_loss=cp.new_zeros(())
                if ARM=='pair':
                    rp=branch_score(model,rejected,n,stops)
                    pair_loss=F.relu(args.trajectory_margin-(cp-rp))
                repair=-cp+pair_loss
                record.update(action=spec['action'],chosen_logp=float(cp.detach()),
                              pair_loss=float(pair_loss.detach()))
                value=float(repair.detach());loss_value+=args.trajectory_alpha*value
                (args.trajectory_alpha*repair/args.grad_accum).backward()
                del cp,pair_loss,repair
                if ARM=='pair':del rp
            count+=1
            if count%args.grad_accum==0 or count==len(loader)*args.epochs or (args.max_steps and count>=args.max_steps):
                last_grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                if not torch.isfinite(torch.tensor(last_grad)):raise ValueError('Nonfinite gradient')
                opt.step();opt.zero_grad(set_to_none=True)
            if count==1 or count%args.log_every==0 or count==len(loader)*args.epochs:
                record.update(step=count,loss=loss_value,gradient_norm=last_grad,active_pairs=active,
                              peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                history.append(record);print(json.dumps(record),flush=True)
            if args.max_steps and count>=args.max_steps:break
        if args.max_steps and count>=args.max_steps:break
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'completion_training_audit.json').write_text(json.dumps({'arm':ARM,'seen_rows':count,'active_pairs':active,
        'action_counts':action_counts,'eos_ids':stops,'kl_scope':'first unique citation-supported units anywhere in the trajectory',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    return history


def skipped(args,model,tokenizer,rows,out_dir):
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    (Path(out_dir)/'raw_eval_skipped.json').write_text(json.dumps({'reason':'Frozen interpolation=0.5 is the sole evaluated endpoint; no redundant raw generation.'}))


legacy.train=train
legacy.evaluate=skipped
if __name__=='__main__':legacy.main()

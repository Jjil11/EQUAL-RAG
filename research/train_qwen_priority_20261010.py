"""Qwen-only controlled repair pilot; unconstrained inference is unchanged."""
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
from train_completion_repair_20261010 import branch_score, skipped

ARM=os.environ['EQUAL_QWEN_PRIORITY_ARM']
assert ARM in {'seq_full','seq_local','action_pair','action_ce'}


def action_loss(logits, allowed, rejected, pair, margin):
    lp=F.log_softmax(logits.float(),-1)
    positive=lp[allowed]
    ce=-torch.logsumexp(positive,0)
    hinge=F.relu(margin-(positive.max()-lp[rejected])) if pair else ce.new_zeros(())
    return ce+hinge,ce,hinge


def train(args,model,tokenizer,rows,anchor_model=None):
    assert args.method=='trajectory_repair' and args.pref_alpha==args.stop_alpha==args.dual_max==0
    loader=DataLoader(legacy.Rows(rows),batch_size=1,shuffle=True,collate_fn=legacy.one)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    model.train();model.config.use_cache=False;opt.zero_grad(set_to_none=True)
    device=next(model.parameters()).device;stops=eos_ids(tokenizer)
    history=[];count=0;active=0;retained=0;last_grad=None
    scope='full_retention_mask' if ARM=='seq_full' else 'failure_local_retention_mask'
    for epoch in range(args.epochs):
        for row in loader:
            mask=torch.tensor(row[scope],device=device,dtype=torch.bool)
            retained+=int(mask.sum());loss_value=0.;kl_value=0.
            if mask.any():
                _,_,_,student=legacy.response_logp(model,tokenizer,row['prompt'],row['short'],args.max_length,True)
                anchor=torch.load(legacy.anchor_cache_path(args.anchor_logits_dir,row['id']),map_location=device,weights_only=True)
                assert student.shape==anchor.shape and len(mask)==len(student)
                a=F.log_softmax(anchor[mask].float(),-1);s=F.log_softmax(student[mask].float(),-1)
                kl=(a.exp()*(a-s)).sum(-1).mean()
                safety=args.anchor_alpha*F.relu(kl-args.anchor_kl_target)
                kl_value=float(kl.detach());loss_value=float(safety.detach())
                (safety/args.grad_accum).backward()
                del student,anchor,a,s,kl,safety
            record={'id':row['id'],'arm':ARM,'anchor_kl':kl_value,'retained_tokens':int(mask.sum())}
            if row.get('has_trajectory'):
                active+=1
                if ARM.startswith('seq_'):
                    chosen,rejected,n,spec=compiled_pair(tokenizer,row,stops,args.max_length)
                    assert spec==row['completion_pair']
                    cp=branch_score(model,chosen,n,stops);rp=branch_score(model,rejected,n,stops)
                    hinge=F.relu(args.trajectory_margin-(cp-rp));ce=-cp;repair=ce+hinge
                else:
                    spec=row['action_decision']
                    ids=torch.tensor(spec['context_ids'],device=device)[None]
                    logits=model(input_ids=ids).logits[0,-1]
                    repair,ce,hinge=action_loss(logits,spec['allowed_ids'],spec['rejected_id'],
                                              ARM=='action_pair',args.trajectory_margin)
                record.update(ce=float(ce.detach()),hinge=float(hinge.detach()))
                loss_value+=args.trajectory_alpha*float(repair.detach())
                (args.trajectory_alpha*repair/args.grad_accum).backward()
                del repair,ce,hinge
                if ARM.startswith('seq_'):del cp,rp
                else:del logits,ids
            count+=1
            if count%args.grad_accum==0 or count==len(loader)*args.epochs or (args.max_steps and count>=args.max_steps):
                last_grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
                assert torch.isfinite(torch.tensor(last_grad))
                opt.step();opt.zero_grad(set_to_none=True)
            if count==1 or count%args.log_every==0 or count==len(loader)*args.epochs:
                record.update(step=count,loss=loss_value,gradient_norm=last_grad,active_pairs=active,
                              peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                history.append(record);print(json.dumps(record),flush=True)
            if args.max_steps and count>=args.max_steps:break
        if args.max_steps and count>=args.max_steps:break
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'priority_training_audit.json').write_text(json.dumps({'arm':ARM,'seen_rows':count,
        'active_pairs':active,'kl_scope':scope,'retained_tokens':retained,'eos_ids':stops,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    return history


legacy.train=train
legacy.evaluate=skipped
if __name__=='__main__':legacy.main()

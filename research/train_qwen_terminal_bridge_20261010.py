"""Earliest-action repair plus a separately supervised verified terminal site."""
import hashlib,json,os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import train_trajectory_equal as legacy
from train_completion_repair_20261010 import skipped
os.environ['EQUAL_QWEN_PRIORITY_ARM']='action_pair'
from train_qwen_priority_20261010 import action_loss
from compile_qwen_terminal_bridge_20261010 import RUN,CACHE

ARM=os.environ['EQUAL_QWEN_TERMINAL_ARM']
assert ARM in {'action_pair','action_ce','action_ul'}

def terminal_loss(logits,allowed):
    return -torch.logsumexp(F.log_softmax(logits.float(),-1)[allowed],0)

def first_loss(logits,allowed,rejected):
    if ARM!='action_ul':return action_loss(logits,allowed,rejected,ARM=='action_pair',1.)
    lp=F.log_softmax(logits.float(),-1);ce=-torch.logsumexp(lp[allowed],0)
    ul=-torch.log1p(-lp[rejected].exp().clamp(max=1-1e-7));return ce+ul,ce,ul

def train(args,model,tokenizer,rows,anchor_model=None):
    assert args.method=='trajectory_repair' and args.trajectory_alpha==.1
    assert args.seed==20260930 and args.epochs==1 and args.lr==1e-5 and Path(args.anchor_logits_dir)==CACHE
    assert args.stop_alpha==args.pref_alpha==args.dual_max==0
    audit=json.loads((RUN/'compiled/audit.json').read_text())
    assert hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest()==audit['compiled_sha256']
    loader=DataLoader(legacy.Rows(rows),batch_size=1,shuffle=True,collate_fn=legacy.one)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    model.train();model.config.use_cache=False;opt.zero_grad(set_to_none=True)
    device=next(model.parameters()).device;history=[];count=active=terminal=0;grad=None
    actions={'stop':0,'complete_then_stop':0}
    for row in loader:
        meta=row['cache_metadata']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==meta['prompt_sha256']
        assert hashlib.sha256(row['short'].encode()).hexdigest()==meta['response_sha256']
        mask=torch.tensor(row['failure_local_retention_mask'],device=device,dtype=torch.bool)
        value=kl_value=0.;record={'id':row['id'],'arm':ARM}
        if mask.any():
            _,_,_,student=legacy.response_logp(model,tokenizer,row['prompt'],row['short'],args.max_length,True)
            anchor=torch.load(legacy.anchor_cache_path(args.anchor_logits_dir,row['id']),map_location=device,weights_only=True)
            assert student.shape==anchor.shape and len(mask)==len(student)
            a=F.log_softmax(anchor[mask].float(),-1);s=F.log_softmax(student[mask].float(),-1)
            kl=(a.exp()*(a-s)).sum(-1).mean();safety=args.anchor_alpha*F.relu(kl-args.anchor_kl_target)
            kl_value=float(kl.detach());value=float(safety.detach());(safety/args.grad_accum).backward()
            del student,anchor,a,s,kl,safety
        if row['has_trajectory']:
            active+=1;actions[row['completion_pair']['action']]+=1
            d=row['action_decision'];ids=torch.tensor(d['context_ids'],device=device)[None]
            logits=model(input_ids=ids).logits[0,-1];repair,ce,contrast=first_loss(logits,d['allowed_ids'],d['rejected_id'])
            record.update(ce=float(ce.detach()),contrast=float(contrast.detach()))
            value+=args.trajectory_alpha*float(repair.detach());(args.trajectory_alpha*repair/args.grad_accum).backward()
            del logits,ids,repair,ce,contrast
            d=row['terminal_decision'];ids=torch.tensor(d['context_ids'],device=device)[None]
            logits=model(input_ids=ids).logits[0,-1];eos=terminal_loss(logits,d['allowed_ids'])
            record['terminal_ce']=float(eos.detach());value+=.1*float(eos.detach())
            (.1*eos/args.grad_accum).backward();terminal+=1;del logits,ids,eos
        count+=1
        if count%args.grad_accum==0 or count==len(loader):
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert torch.isfinite(torch.tensor(grad))
            opt.step();opt.zero_grad(set_to_none=True)
        if count==1 or count%args.log_every==0 or count==len(loader):
            record.update(step=count,loss=value,anchor_kl=kl_value,active_pairs=active,terminal_sites=terminal,
                gradient_norm=grad,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
            history.append(record);print(json.dumps(record),flush=True)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'terminal_training_audit.json').write_text(json.dumps({'arm':ARM,'seen_rows':count,'active_pairs':active,
        'terminal_sites':terminal,'action_counts':actions,'compiled_data_sha256':audit['compiled_sha256'],
        'first_action_alpha':.1,'terminal_ce_alpha':.1,'repair_forwards_per_active_row':2,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'warning':'Added EOS objective and extra forward relative to earlier round; not isolated equal-total-strength ablation.'},indent=2))
    return history

legacy.train=train;legacy.evaluate=skipped
if __name__=='__main__':legacy.main()

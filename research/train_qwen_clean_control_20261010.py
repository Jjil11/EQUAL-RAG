"""Supplemental fixed strong-CE and matched next-action UL controls."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
os.environ['EQUAL_QWEN_CLEAN_ARM']='action_ce'
import train_qwen_clean_20261010 as core

KIND=os.environ['EQUAL_QWEN_CLEAN_CONTROL']
assert KIND in {'strong_ce','action_ul'}
original_train=core.train


def matched_ul(logits,allowed,rejected,pair,margin):
    lp=F.log_softmax(logits.float(),-1)
    ce=-torch.logsumexp(lp[allowed],0)
    ul=-torch.log1p(-lp[rejected].exp().clamp(max=1-1e-7))
    return ce+ul,ce,ul


if KIND=='action_ul':core.action_loss=matched_ul


def train(args,model,tokenizer,rows,anchor_model=None):
    assert args.trajectory_alpha==(.2 if KIND=='strong_ce' else .1)
    history=original_train(args,model,tokenizer,rows,anchor_model)
    out=Path(args.out_dir)
    (out/'supplemental_control_manifest.json').write_text(json.dumps({'control':KIND,
        'repair_alpha':args.trajectory_alpha,'seed':args.seed,'training_rows':len(rows),
        'same_clean_labels_and_failure_local_kl':True,'preference_margin_not_used':True,
        'logged_hinge_field_is':'unlikelihood' if KIND=='action_ul' else 'zero',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'warning':'Strong CE alpha=0.2 is a finite strength control, not exact gradient norm matching; UL uses the same earliest invalid token and one repair forward.'},indent=2))
    return history


core.legacy.train=train
if __name__=='__main__':core.legacy.main()

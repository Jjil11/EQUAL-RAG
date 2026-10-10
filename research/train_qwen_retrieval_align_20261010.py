"""Same action repairs and KL on newly frozen raw-top5 training trajectories."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from build_qwen_retrieval_align_20261010 import RUN,CACHE

ARM=os.environ['EQUAL_QWEN_RETRIEVAL_ARM']
assert ARM in {'action_pair','action_ce','action_ul','strong_ce'}
os.environ['EQUAL_QWEN_CLEAN_ARM']='action_pair' if ARM=='action_pair' else 'action_ce'
import train_qwen_clean_20261010 as core

original_train=core.train


def ul_loss(logits,allowed,rejected,pair,margin):
    lp=F.log_softmax(logits.float(),-1);ce=-torch.logsumexp(lp[allowed],0)
    ul=-torch.log1p(-lp[rejected].exp().clamp(max=1-1e-7))
    return ce+ul,ce,ul


if ARM=='action_ul':core.action_loss=ul_loss


def train(args,model,tokenizer,rows,anchor_model=None):
    assert args.trajectory_alpha==(.2 if ARM=='strong_ce' else .1)
    assert Path(args.anchor_logits_dir)==CACHE and args.seed==20260930 and args.epochs==1
    audit=json.loads((RUN/'compiled/audit.json').read_text())
    assert hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest()==audit['compiled_sha256']
    for row in rows:
        meta=row['cache_metadata'];p=core.legacy.anchor_cache_path(args.anchor_logits_dir,row['id'])
        assert str(p)==meta['path'] and p.exists()
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==meta['prompt_sha256']
        assert hashlib.sha256(row['short'].encode()).hexdigest()==meta['response_sha256']
        assert len(row['failure_local_retention_mask'])==meta['shape'][0]
    history=original_train(args,model,tokenizer,rows,anchor_model)
    out=Path(args.out_dir)
    (out/'retrieval_alignment_manifest.json').write_text(json.dumps({'arm':ARM,'repair_alpha':args.trajectory_alpha,
        'compiled_data_sha256':audit['compiled_sha256'],'same_top5_train_labels_and_kl':True,
        'all_350_rows_retained':len(rows)==350,'active_rows':sum(r['has_trajectory'] for r in rows),
        'logged_hinge_field_is':'unlikelihood' if ARM=='action_ul' else 'margin_hinge' if ARM=='action_pair' else 'zero',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'warning':'No inference constraints; all arms one repair forward, not gradient-strength matched; unknown zero-edge repetitions are not given forced stop labels.'},indent=2))
    return history


core.legacy.train=train
if __name__=='__main__':core.legacy.main()

"""Unchanged optimizer/budget; first stop and EOS contexts jointly aligned."""
import hashlib,json
from pathlib import Path
import train_qwen_terminal_bridge_20261010 as core
from compile_qwen_native_consistent_20261010 import RUN,SOURCE
core.RUN=RUN
original=core.train

def train(args,model,tokenizer,rows,anchor_model=None):
    history=original(args,model,tokenizer,rows,anchor_model)
    out=Path(args.out_dir)
    (out/'native_training_manifest.json').write_text(json.dumps({'arm':core.ARM,
        'seen_rows':len(rows),'native_eos_sites':sum(r.get('terminal_decision',{}).get('origin')=='recorded_onpolicy_prefix' for r in rows),
        'same_loss_weights_and_forward_budget_as_bridge':True,'parent_round':str(SOURCE),
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'core_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()},indent=2))
    return history

core.legacy.train=train
if __name__=='__main__':core.legacy.main()

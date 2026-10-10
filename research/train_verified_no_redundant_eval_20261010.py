"""Same frozen repair training; skip unused raw eval after saving KL-only weights."""
import json
from pathlib import Path
import train_verified_boundary_20261010 as verified


def skipped(args,model,tokenizer,rows,out_dir):
    if verified.ARM!='kl_only':raise ValueError('Skip wrapper only allowed for KL-only control')
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    (Path(out_dir)/'raw_evaluation_skipped.json').write_text(json.dumps({
        'reason':'Unused un-interpolated development endpoint; final 0.5 adapter evaluation is unchanged.',
        'model_training_unchanged':True,'heldout_predictions':0},indent=2))
    print('Saved KL-only adapter; unused raw eval skipped',flush=True)


verified.legacy.evaluate=skipped
if __name__=='__main__':verified.legacy.main()

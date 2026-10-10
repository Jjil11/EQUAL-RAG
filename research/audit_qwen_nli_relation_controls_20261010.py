"""Separate adversarial relation sanity, never tunes the fixed judge."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
ROOT=Path(str(_release_root));OUT=ROOT/'runs/qwen_nli_relation_controls_20261010'
assert not OUT.exists();OUT.mkdir()
path=Path(str(_release_models / 'DeBERTa-v3-base-mnli-fever-anli'))
from verify_nli_snapshot_20261010 import verify
verify(path)
tok=AutoTokenizer.from_pretrained(path);m=AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()
cases=[('Alice wrote the screenplay. Bob directed the movie.','Who directed the movie? Alice.',0),
 ('Alice wrote the screenplay. Bob directed the movie.','Who directed the movie? Bob.',1),
 ('The Tigers lost the final to the Red Sox. The Red Sox won the championship.','Which team won the final? Tigers.',0),
 ('The Tigers lost the final to the Red Sox. The Red Sox won the championship.','Which team won the final? Red Sox.',1),
 ('Alice was born in London. Bob was born in Paris.','Where was Alice born? Paris.',0),
 ('Alice was born in London. Bob was born in Paris.','Where was Alice born? London.',1),
 ('Alice directed Film X. Bob wrote Film X. Carol directed Film Y. Dan wrote Film Y.','Who directed the film written by Bob? Carol.',0),
 ('Alice directed Film X. Bob wrote Film X. Carol directed Film Y. Dan wrote Film Y.','Who directed the film written by Bob? Alice.',1)]
values=[]
for p,h,y in cases:
 inp=tok(p,h,return_tensors='pt').to('cuda')
 with torch.inference_mode():prob=m(**inp).logits.float().softmax(-1)[0].cpu().tolist()
 values.append({'premise':p,'hypothesis':h,'expected':y,'observed':int(max(range(3),key=lambda i:prob[i])==0),'probabilities':prob})
report={'status':'COMPLETE_ADVERSARIAL_RELATION_SANITY','revision':'6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
 'n':8,'passed':sum(v['expected']==v['observed'] for v in values),'cases':values,
 'warning':'Constructed software challenge, not human evaluation or calibrated accuracy; no threshold change regardless of outcome.'}
report['all_passed']=report['passed']==8
(OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

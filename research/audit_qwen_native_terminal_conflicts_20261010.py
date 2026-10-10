"""Read-only training-label consistency, before accepting a native pilot."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
from transformers import AutoTokenizer
from compile_qwen_native_terminal_20261010 import ROOT,RUN
tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
rows=[json.loads(s) for s in (RUN/'compiled/train.jsonl').read_text().splitlines() if s.strip()]
records=[]
for r in rows:
    if not r['has_trajectory']:continue
    first=r['action_decision'];terminal=r['terminal_decision'];positive=terminal['context_ids']+[terminal['canonical_eos_id']]
    c=first['context_ids'];bad=first['rejected_id']
    exact=positive[:len(c)]==c and len(positive)>len(c) and positive[len(c)]==bad
    text=tok.decode(c+[bad],skip_special_tokens=True)
    semantic=tok.decode(terminal['context_ids'],skip_special_tokens=True).startswith(text)
    if exact or semantic:records.append({'id':r['id'],'origin':terminal['origin'],'exact_token_path_conflict':exact,'decoded_prefix_conflict':semantic})
print(json.dumps({'rows':350,'active':150,'exact_conflicts':sum(r['exact_token_path_conflict'] for r in records),
    'decoded_prefix_conflicts':sum(r['decoded_prefix_conflict'] for r in records),'examples':records[:6]},indent=2),flush=True)

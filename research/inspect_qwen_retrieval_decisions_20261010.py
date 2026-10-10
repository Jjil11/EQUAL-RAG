"""Read-only train-label token boundary diagnosis."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
from pathlib import Path
from transformers import AutoTokenizer
from build_qwen_retrieval_align_20261010 import RUN
tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
rows=[json.loads(s) for s in (RUN/'compiled/train.jsonl').read_text().splitlines() if s.strip()]
for action in ['stop','complete_then_stop']:
    selected=[r for r in rows if r['has_trajectory'] and r['completion_pair']['action']==action][:4]
    for r in selected:
        d=r['action_decision'];c=r['completion_pair']
        print(json.dumps({'id':r['id'],'action':action,'context_tail':tok.decode(d['context_ids'][-30:]),
            'allowed_tokens':[(a,tok.decode([a])) for a in d['allowed_ids']],
            'rejected_token':(d['rejected_id'],tok.decode([d['rejected_id']])),
            'chosen_tail':c['chosen'][-180:],'chosen_suffix_tokens':c['chosen_tokens']},ensure_ascii=False),flush=True)

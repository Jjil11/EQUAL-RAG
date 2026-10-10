#!/usr/bin/env python3
"""Check whether repeated Qwen3+PEFT loads agree on one physical GPU."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models

import gc
import json

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_equal_rag_pilot import encode_example

MODEL = str(_release_model)
ADAPTER = str(_release_root / 'runs/qwen3_14b_aequal/short_stop_seed20260930/adapter')
DATA = str(_release_root / 'data/qampari_evidence_full/train.jsonl')

with open(DATA, encoding="utf-8") as handle:
    row = json.loads(next(handle))
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tokenizer._equal_rag_disable_thinking = True
ids, mask, labels = encode_example(tokenizer, row["prompt"], row["short"], 2048)
valid = labels[1:].ne(-100).cuda()


def run_once():
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER).cuda().eval()
    with torch.inference_mode():
        logits = model(input_ids=ids[None].cuda(), attention_mask=mask[None].cuda()).logits[:, :-1]
    selected = logits[0, valid].float().cpu()
    del logits, model, base
    gc.collect(); torch.cuda.empty_cache()
    return selected


l0, l1 = run_once(), run_once()
p0, p1 = F.log_softmax(l0, -1), F.log_softmax(l1, -1)
kl = (p1.exp() * (p1 - p0)).sum(-1)
print(json.dumps({
    "max_abs_diff": float((l0 - l1).abs().max()),
    "mean_abs_diff": float((l0 - l1).abs().mean()),
    "kl_mean": float(kl.mean()),
    "kl_max": float(kl.max()),
}, indent=2))

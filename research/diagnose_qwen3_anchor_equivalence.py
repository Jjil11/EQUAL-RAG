#!/usr/bin/env python3
"""Compare two separately loaded Qwen3+PEFT copies before any update."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models

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


def load(device: str):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model = PeftModel.from_pretrained(base, ADAPTER).to(device).eval()
    return model


m0, m1 = load("cuda:0"), load("cuda:1")
with torch.inference_mode():
    out0 = m0(input_ids=ids[None].to("cuda:0"), attention_mask=mask[None].to("cuda:0")).logits[:, :-1]
    out1 = m1(input_ids=ids[None].to("cuda:1"), attention_mask=mask[None].to("cuda:1")).logits[:, :-1]
    with m0.disable_adapter(), m1.disable_adapter():
        base0 = m0(input_ids=ids[None].to("cuda:0"), attention_mask=mask[None].to("cuda:0")).logits[:, :-1]
        base1 = m1(input_ids=ids[None].to("cuda:1"), attention_mask=mask[None].to("cuda:1")).logits[:, :-1]
valid = labels[1:].ne(-100)
l0 = out0[0, valid.to("cuda:0")].float()
l1 = out1[0, valid.to("cuda:1")].float().to("cuda:0")
p1 = F.log_softmax(l1, dim=-1)
p0 = F.log_softmax(l0, dim=-1)
kl = (p1.exp() * (p1 - p0)).sum(-1)
bl0 = base0[0, valid.to("cuda:0")].float()
bl1 = base1[0, valid.to("cuda:1")].float().to("cuda:0")
bp1 = F.log_softmax(bl1, dim=-1)
bp0 = F.log_softmax(bl0, dim=-1)
base_kl = (bp1.exp() * (bp1 - bp0)).sum(-1)


def adapter_checksum(model) -> dict:
    tensors = [p.detach().float() for n, p in model.named_parameters() if "lora_" in n]
    return {
        "count": len(tensors),
        "sum": sum(float(t.sum().cpu()) for t in tensors),
        "abs_sum": sum(float(t.abs().sum().cpu()) for t in tensors),
    }


def parameter_samples(model) -> dict:
    wanted = (
        "embed_tokens.weight", "layers.0.self_attn.q_proj.base_layer.weight",
        "layers.39.mlp.down_proj.base_layer.weight", "norm.weight", "lm_head.weight",
    )
    result = {}
    for name, parameter in model.named_parameters():
        if any(name.endswith(suffix) for suffix in wanted):
            flat = parameter.detach().float().flatten()
            result[name] = {
                "shape": list(parameter.shape),
                "head_sum": float(flat[:4096].sum().cpu()),
                "head_abs_sum": float(flat[:4096].abs().sum().cpu()),
            }
    return result


print(json.dumps({
    "response_positions": int(valid.sum()),
    "adapter0": adapter_checksum(m0),
    "adapter1": adapter_checksum(m1),
    "parameter_samples0": parameter_samples(m0),
    "parameter_samples1": parameter_samples(m1),
    "logit_max_abs_diff": float((l0 - l1).abs().max().cpu()),
    "logit_mean_abs_diff": float((l0 - l1).abs().mean().cpu()),
    "kl_mean": float(kl.mean().cpu()),
    "kl_max": float(kl.max().cpu()),
    "base_logit_max_abs_diff": float((bl0 - bl1).abs().max().cpu()),
    "base_logit_mean_abs_diff": float((bl0 - bl1).abs().mean().cpu()),
    "base_kl_mean": float(base_kl.mean().cpu()),
    "base_kl_max": float(base_kl.max().cpu()),
}, indent=2))

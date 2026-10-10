#!/usr/bin/env python3
"""Cache frozen-anchor response logits for deterministic one-GPU A-EQUAL-KL."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_equal_rag_pilot import anchor_cache_path, read_jsonl, response_logp, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--disable-thinking", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    rows = read_jsonl(args.train)
    if args.max_train:
        rows = rows[: args.max_train]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    tokenizer._equal_rag_disable_thinking = args.disable_thinking
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter).cuda()
    model.eval()
    model.config.use_cache = False

    started = time.time()
    total_tokens = 0
    with torch.inference_mode():
        for idx, row in enumerate(rows, 1):
            _, _, _, logits = response_logp(
                model, tokenizer, row["prompt"], row["short"], args.max_length,
                return_response_logits=True,
            )
            logits = logits.to(dtype=torch.bfloat16, device="cpu").contiguous()
            torch.save(logits, anchor_cache_path(str(out_dir), row["id"]))
            total_tokens += logits.shape[0]
            if idx % 20 == 0 or idx == len(rows):
                print(f"cached {idx}/{len(rows)} response_tokens={total_tokens}", flush=True)

    manifest = vars(args) | {
        "examples": len(rows),
        "response_tokens": total_tokens,
        "dtype": "bfloat16",
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interleaved paired latency benchmark for two LoRA adapters on one model."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-adapter", required=True)
    parser.add_argument("--method-adapter", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.baseline_adapter, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.baseline_adapter, adapter_name="baseline")
    model.load_adapter(args.method_adapter, adapter_name="method")
    model = model.cuda().eval()
    model.config.use_cache = True
    order_rng = random.Random(args.seed)
    records = []
    for index, row in enumerate(rows):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=not args.disable_thinking,
        )
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=args.max_length).to("cuda")
        order = ["baseline", "method"]
        if order_rng.random() < 0.5:
            order.reverse()
        sample = {"id": row["id"], "warmup": index < args.warmup, "order": order}
        for adapter in order:
            model.set_adapter(adapter)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            sample[adapter] = {
                "seconds": time.perf_counter() - start,
                "generated_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
            }
        records.append(sample)
        if (index + 1) % 20 == 0:
            print(json.dumps({"done": index + 1, "total": len(rows)}), flush=True)
    measured = records[args.warmup:]
    summary = {"n": len(measured), "warmup_excluded": args.warmup}
    for adapter in ("baseline", "method"):
        seconds = [row[adapter]["seconds"] for row in measured]
        tokens = [row[adapter]["generated_tokens"] for row in measured]
        summary[adapter] = {
            "mean_seconds": statistics.fmean(seconds),
            "mean_generated_tokens": statistics.fmean(tokens),
            "total_seconds": sum(seconds), "total_generated_tokens": sum(tokens),
        }
    differences = [row["method"]["seconds"] - row["baseline"]["seconds"] for row in measured]
    summary["raw_paired_mean_seconds_delta"] = statistics.fmean(differences)
    summary["records"] = records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure actual autoregressive generation latency and generated token count."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter).cuda().eval()
    model.config.use_cache = True
    records = []
    for index, row in enumerate(rows):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=not args.disable_thinking,
        )
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=args.max_length).to("cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = int(generated.shape[1] - inputs["input_ids"].shape[1])
        records.append({"id": row["id"], "seconds": elapsed, "generated_tokens": tokens, "warmup": index < args.warmup})
        if (index + 1) % 20 == 0:
            print(json.dumps({"done": index + 1, "total": len(rows)}), flush=True)
    measured = records[args.warmup:]
    seconds = [row["seconds"] for row in measured]
    tokens = [row["generated_tokens"] for row in measured]
    report = {
        "n": len(measured), "warmup_excluded": args.warmup,
        "mean_seconds": statistics.fmean(seconds), "median_seconds": statistics.median(seconds),
        "p95_seconds": percentile(seconds, 0.95), "total_seconds": sum(seconds),
        "mean_generated_tokens": statistics.fmean(tokens), "total_generated_tokens": sum(tokens),
        "tokens_per_second": sum(tokens) / sum(seconds), "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

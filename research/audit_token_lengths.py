#!/usr/bin/env python3
"""Audit prompt/response lengths before launching an EQUAL-RAG dataset."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--generation-cap", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    report = {}
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in Path(args.data, f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        totals, responses = [], []
        for row in rows:
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["short"]},
            ]
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            totals.append(len(tokenizer(full, add_special_tokens=False)["input_ids"]))
            responses.append(len(tokenizer(row["short"], add_special_tokens=False)["input_ids"]))
        report[split] = {
            "n": len(rows),
            "total_mean": statistics.fmean(totals),
            "total_p95": percentile(totals, 0.95),
            "total_max": max(totals),
            "over_max_length": sum(value > args.max_length for value in totals),
            "response_mean": statistics.fmean(responses),
            "response_p95": percentile(responses, 0.95),
            "response_max": max(responses),
            "response_over_generation_cap": sum(value > args.generation_cap for value in responses),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

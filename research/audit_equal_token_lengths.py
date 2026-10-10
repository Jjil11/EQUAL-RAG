#!/usr/bin/env python3
"""Audit prompt/target lengths for one or more local tokenizers."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def stats(values):
    ordered = sorted(values)
    return {"mean": statistics.fmean(values), "p95": ordered[int(0.95 * (len(ordered) - 1))], "max": max(values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {}
    for label, path in args.tokenizer:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        split_report = {}
        for split in ("train", "dev", "test"):
            rows = [json.loads(x) for x in (args.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if x]
            prompts, targets, totals = [], [], []
            for row in rows:
                prefix = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False,
                                                 add_generation_prompt=True)
                p_len = len(tok(prefix, add_special_tokens=False)["input_ids"])
                t_len = len(tok(row["short"], add_special_tokens=False)["input_ids"])
                prompts.append(p_len); targets.append(t_len); totals.append(p_len + t_len)
            split_report[split] = {"n": len(rows), "prompt": stats(prompts), "target": stats(targets), "total": stats(totals),
                                   "over_2048": sum(x > 2048 for x in totals), "target_over_256": sum(x > 256 for x in targets)}
        report[label] = split_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a leakage-safe, input-conditioned token-budget prompting baseline.

The budget rule is calibrated only on training targets.  At evaluation time it
uses the number of retrieved documents, which is observable from the input;
held-out answer labels never enter the rule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_text(row: dict) -> str:
    for key in ("short", "target", "answer"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key]
    raise ValueError(f"No target text for row {row.get('id')}")


def add_budget(prompt: str, budget: int) -> str:
    instruction = (
        f"\n\nToken budget: Give a complete answer using at most {budget} output tokens."
    )
    marker = "\nAnswer:"
    if prompt.rstrip().endswith("Answer:"):
        return prompt.rstrip()[: -len("Answer:")].rstrip() + instruction + marker
    return prompt.rstrip() + instruction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-budget", type=int, default=32)
    parser.add_argument("--max-budget", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train = read_jsonl(Path(args.train))
    ratios = []
    target_lengths = []
    for row in train:
        length = len(tokenizer(target_text(row), add_special_tokens=False).input_ids)
        docs = max(1, len(row.get("docs", [])))
        target_lengths.append(length)
        ratios.append(length / docs)
    tokens_per_doc = float(np.quantile(ratios, args.quantile, method="higher"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for input_name in args.inputs:
        source = Path(input_name)
        rows = read_jsonl(source)
        budgets = []
        with (out_dir / source.name).open("w", encoding="utf-8") as handle:
            for original in rows:
                row = dict(original)
                docs = max(1, len(row.get("docs", [])))
                budget = min(args.max_budget, max(args.min_budget, math.ceil(tokens_per_doc * docs)))
                row["prompt"] = add_budget(row["prompt"], budget)
                row["dynamic_token_budget"] = budget
                row["dynamic_budget_observable"] = {"retrieved_document_count": docs}
                budgets.append(budget)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_stats[source.stem] = {
            "n": len(rows),
            "mean_budget": sum(budgets) / max(1, len(budgets)),
            "min_budget": min(budgets),
            "max_budget": max(budgets),
        }

    manifest = {
        "protocol": "training-only calibration; held-out labels unused",
        "model": args.model,
        "quantile": args.quantile,
        "rule": "ceil(q95(training target tokens / retrieved docs) * input retrieved docs)",
        "tokens_per_doc": tokens_per_doc,
        "train_target_tokens": {
            "n": len(target_lengths),
            "mean": sum(target_lengths) / len(target_lengths),
            "min": min(target_lengths),
            "max": max(target_lengths),
        },
        "bounds": [args.min_budget, args.max_budget],
        "splits": split_stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

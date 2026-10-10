#!/usr/bin/env python3
"""Compare redundancy-only training against controls on both base models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--mistral-controls", type=Path, required=True)
    parser.add_argument("--qwen-controls", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    report = {"bootstrap_samples": args.samples, "models": {}}
    for offset, (model, controls) in enumerate(
        (("mistral", args.mistral_controls), ("qwen", args.qwen_controls))
    ):
        redundant = read_jsonl(args.ablation / model / "eval_results.jsonl")
        short = read_jsonl(controls / "short_stop" / "eval_results.jsonl")
        full = read_jsonl(controls / "equal_full_stop" / "eval_results.jsonl")
        report["models"][model] = {
            "truncation_rate_at_128": {
                "redundancy_stop": truncation_rate(redundant),
                "short_stop": truncation_rate(short),
                "equal_full_stop": truncation_rate(full),
            },
            "redundancy_stop_vs_short_stop": paired_bootstrap(
                redundant, short, args.seed + offset * 10, args.samples
            ),
            "redundancy_stop_vs_equal_full_stop": paired_bootstrap(
                redundant, full, args.seed + offset * 10 + 1, args.samples
            ),
        }

    output = args.ablation / "paired_analysis.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

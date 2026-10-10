#!/usr/bin/env python3
"""Paired analysis of C-EQUAL against same-seed Mistral controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cequal", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    adaptive = read_jsonl(args.cequal / "eval_results.jsonl")
    short = read_jsonl(args.controls / "short_stop" / "eval_results.jsonl")
    fixed = read_jsonl(args.controls / "equal_full_stop" / "eval_results.jsonl")
    report = {
        "truncation_rate_at_128": {
            "cequal_dual": truncation_rate(adaptive),
            "short_stop": truncation_rate(short),
            "equal_full_stop": truncation_rate(fixed),
        },
        "cequal_dual_vs_short_stop": paired_bootstrap(
            adaptive, short, 20261002, args.samples
        ),
        "cequal_dual_vs_equal_full_stop": paired_bootstrap(
            adaptive, fixed, 20261003, args.samples
        ),
    }
    output = args.cequal / "paired_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

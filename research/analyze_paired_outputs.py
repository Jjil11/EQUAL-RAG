#!/usr/bin/env python3
"""Generic paired bootstrap analysis for two EQUAL-RAG output files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_qwen3_cached_aequal import utility_bootstrap
from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--truncation-cap", type=int, default=128)
    args = ap.parse_args()

    method = read_jsonl(args.method)
    baseline = read_jsonl(args.baseline)
    report = {
        "paired_method_minus_baseline": paired_bootstrap(
            method, baseline, args.seed, args.samples
        ),
        "grounded_utility_per_token": utility_bootstrap(
            method, baseline, args.seed + 1, args.samples
        ),
        f"truncation_rate_{args.truncation_cap}": {
            "method": truncation_rate(method, args.truncation_cap),
            "baseline": truncation_rate(baseline, args.truncation_cap),
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

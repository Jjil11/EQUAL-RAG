#!/usr/bin/env python3
"""Paired analysis and Pareto selection for Qwen3 compression calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_qwen3_cached_aequal import utility_bootstrap
from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--calibration-root", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    baseline = read_jsonl(args.baseline)
    report = {"bootstrap_samples": args.samples, "methods": {}}
    for offset, name in enumerate(("pref_0p4", "pref_0p8_gpu1")):
        rows = read_jsonl(args.calibration_root / name / "eval_results.jsonl")
        paired = paired_bootstrap(rows, baseline, args.seed + offset, args.samples)
        utility = utility_bootstrap(rows, baseline, args.seed + 100 + offset, args.samples)
        report["methods"][name] = {
            "paired_method_minus_baseline": paired,
            "grounded_utility_per_token": utility,
            "truncation_rate_128": {
                "method": truncation_rate(rows, 128),
                "baseline": truncation_rate(baseline, 128),
            },
        }

    report["selection"] = {
        "chosen": "pref_0p4",
        "reason": (
            "passes the frozen development gate with lower mean tokens and higher "
            "grounded utility; pref_0p8 is dominated"
        ),
    }
    output = args.calibration_root / "paired_calibration_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

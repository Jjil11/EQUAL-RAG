#!/usr/bin/env python3
"""Paired bootstrap for the exhaustive eligible QAMPARI holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    full = read_jsonl(args.root / "equal_full_stop" / "eval_results.jsonl")
    short = read_jsonl(args.root / "short_stop" / "eval_results.jsonl")
    report = {
        "truncation_rate_at_128": {
            "equal_full_stop": truncation_rate(full),
            "short_stop": truncation_rate(short),
        },
        "equal_full_stop_vs_short_stop": paired_bootstrap(
            full, short, args.seed, args.samples
        ),
    }
    (args.root / "paired_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

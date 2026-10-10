#!/usr/bin/env python3
"""Frozen dev-only promotion gate for the MultiSpanQA replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.paired.read_text(encoding="utf-8"))
    paired = report["paired_method_minus_baseline"]
    quality = paired["grounded_edge_recall"]
    tokens = paired["output_tokens"]
    utility = report["grounded_utility_per_token"]
    checks = {
        "quality_noninferior_margin_0p01": quality["delta_a_minus_b"] >= -0.01,
        "fewer_generated_tokens": tokens["delta_a_minus_b"] < 0,
        "higher_grounded_utility": utility["relative_delta"] > 0,
    }
    result = {"checks": checks, "passes": all(checks.values()), "statistics": report}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passes"] else 1)


if __name__ == "__main__":
    main()

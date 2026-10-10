#!/usr/bin/env python3
"""Frozen multi-objective dev gate for saturation-aware EQUAL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def repetition(rows: list[dict], cap: int) -> dict:
    duplicate = 0
    cap_hits = 0
    for row in rows:
        parts = [re.sub(r"\s+", " ", x.strip().lower().rstrip(". ;"))
                 for x in row["output"].split(",") if x.strip()]
        duplicate += len(parts) > len(set(parts))
        cap_hits += row["tokens"] >= cap
    n = len(rows)
    return {"any_duplicate_rate": duplicate / n, "cap_hit_rate": cap_hits / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-metrics", type=Path, required=True)
    ap.add_argument("--baseline-results", type=Path, required=True)
    ap.add_argument("--method-metrics", type=Path, required=True)
    ap.add_argument("--method-results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=256)
    args = ap.parse_args()

    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    method = json.loads(args.method_metrics.read_text(encoding="utf-8"))
    baseline_diag = repetition(read_rows(args.baseline_results), args.cap)
    method_diag = repetition(read_rows(args.method_results), args.cap)
    checks = {
        "quality_noninferior": method["grounded_edge_recall"] >= baseline["grounded_edge_recall"] - 0.01,
        "fewer_tokens": method["mean_output_tokens"] < baseline["mean_output_tokens"],
        "higher_utility": method["grounded_utility_per_token"] > baseline["grounded_utility_per_token"],
        "lower_cap_hit_rate": method_diag["cap_hit_rate"] < baseline_diag["cap_hit_rate"],
        "lower_duplicate_rate": method_diag["any_duplicate_rate"] < baseline_diag["any_duplicate_rate"],
    }
    report = {
        "cap": args.cap,
        "quality_floor": baseline["grounded_edge_recall"] - 0.01,
        "baseline": baseline,
        "method": method,
        "baseline_diagnostics": baseline_diag,
        "method_diagnostics": method_diag,
        "checks": checks,
        "passes": all(checks.values()),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passes"] else 1)


if __name__ == "__main__":
    main()

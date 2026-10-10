#!/usr/bin/env python3
"""Predeclared mechanism-stratified analysis for on-policy EQUAL-RAG repair.

Strata are assigned from the *baseline* output only and are mutually exclusive:
cap_hit (tokens >= cap), duplicate_noncap (exact repeated comma-delimited answer
unit without hitting cap), and clean (neither). This prevents the repaired
method's output from deciding which samples are evaluated.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from analyze_qwen3_cached_aequal import utility_bootstrap
from analyze_second_pilot import paired_bootstrap, read_jsonl


def normalize(unit: str) -> str:
    return re.sub(r"\s+", " ", unit.strip().lower().rstrip(". ;"))


def has_duplicate(output: str) -> bool:
    units = [normalize(part) for part in output.split(",")]
    units = [part for part in units if part]
    return len(units) != len(set(units))


def align(method: list[dict], baseline: list[dict]) -> tuple[list[dict], list[dict]]:
    method_by_id = {str(row["id"]): row for row in method}
    baseline_by_id = {str(row["id"]): row for row in baseline}
    if set(method_by_id) != set(baseline_by_id):
        raise ValueError("Evaluation IDs differ between method and baseline")
    ids = sorted(method_by_id)
    return [method_by_id[key] for key in ids], [baseline_by_id[key] for key in ids]


def analyze_pair(label: str, baseline_path: Path, method_path: Path, cap: int, samples: int, seed: int) -> dict:
    method, baseline = align(read_jsonl(method_path), read_jsonl(baseline_path))
    buckets = {"clean": ([], []), "duplicate_noncap": ([], []), "cap_hit": ([], [])}
    for method_row, baseline_row in zip(method, baseline):
        if baseline_row["tokens"] >= cap:
            bucket = "cap_hit"
        elif has_duplicate(baseline_row["output"]):
            bucket = "duplicate_noncap"
        else:
            bucket = "clean"
        buckets[bucket][0].append(method_row)
        buckets[bucket][1].append(baseline_row)

    result = {"label": label, "cap": cap, "baseline": str(baseline_path), "method": str(method_path), "strata": {}}
    for offset, (name, (method_rows, baseline_rows)) in enumerate(buckets.items()):
        if not method_rows:
            result["strata"][name] = {"n": 0}
            continue
        paired = paired_bootstrap(method_rows, baseline_rows, seed + offset, samples)
        utility = utility_bootstrap(method_rows, baseline_rows, seed + 10 + offset, samples)
        result["strata"][name] = {
            "n": len(method_rows),
            "quality_delta": paired["grounded_edge_recall"]["delta_a_minus_b"],
            "quality_ci95": paired["grounded_edge_recall"]["ci95"],
            "tokens_delta": paired["output_tokens"]["delta_a_minus_b"],
            "tokens_ci95": paired["output_tokens"]["ci95"],
            "utility_relative_delta": utility["relative_delta"],
            "utility_ci95": utility["ci95"],
        }
    return result


def aggregate(runs: list[dict]) -> dict:
    result = {}
    for stratum in ("clean", "duplicate_noncap", "cap_hit"):
        entries = [run["strata"][stratum] for run in runs if run["strata"][stratum]["n"]]
        result[stratum] = {"runs": len(entries), "total_examples": sum(x["n"] for x in entries)}
        for metric in ("quality_delta", "tokens_delta", "utility_relative_delta"):
            values = [x[metric] for x in entries]
            result[stratum][metric] = {
                "mean_across_runs": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", nargs=3, metavar=("LABEL", "BASELINE", "METHOD"), required=True)
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [
        analyze_pair(label, Path(baseline), Path(method), args.cap, args.samples, args.seed + 100 * index)
        for index, (label, baseline, method) in enumerate(args.pair)
    ]
    report = {
        "protocol": "Strata assigned solely from baseline outputs; cap_hit takes precedence over duplicate_noncap.",
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

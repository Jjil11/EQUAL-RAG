#!/usr/bin/env python3
"""Paired bootstrap analysis for measured generation latency and token count."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def paired_metric(method: list[dict], baseline: list[dict], field: str, seed: int, samples: int) -> dict:
    method_by_id = {str(row["id"]): row for row in method if not row.get("warmup")}
    baseline_by_id = {str(row["id"]): row for row in baseline if not row.get("warmup")}
    ids = sorted(set(method_by_id) & set(baseline_by_id))
    if len(ids) != len(method_by_id) or len(ids) != len(baseline_by_id):
        raise ValueError("Measured IDs differ")
    differences = [float(method_by_id[key][field]) - float(baseline_by_id[key][field]) for key in ids]
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        boot.append(statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences))
    boot.sort()
    mean_method = statistics.fmean(float(method_by_id[key][field]) for key in ids)
    mean_baseline = statistics.fmean(float(baseline_by_id[key][field]) for key in ids)
    return {
        "n": len(ids), "method": mean_method, "baseline": mean_baseline,
        "delta": mean_method - mean_baseline,
        "relative_delta": mean_method / mean_baseline - 1,
        "ci95": [boot[int(0.025 * samples)], boot[min(samples - 1, int(0.975 * samples))]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", nargs=3, metavar=("LABEL", "BASELINE", "METHOD"), required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for index, (label, baseline_path, method_path) in enumerate(args.pair):
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))["records"]
        method = json.loads(Path(method_path).read_text(encoding="utf-8"))["records"]
        report[label] = {
            "seconds": paired_metric(method, baseline, "seconds", args.seed + 2 * index, args.samples),
            "generated_tokens": paired_metric(method, baseline, "generated_tokens", args.seed + 2 * index + 1, args.samples),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

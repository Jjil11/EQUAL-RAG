#!/usr/bin/env python3
"""Three-seed summary for Mistral short-stop versus EQUAL-full-stop."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


METRIC_KEYS = (
    "answer_recall",
    "citation_coverage",
    "grounded_edge_recall",
    "citation_validity",
    "mean_output_tokens",
    "grounded_utility_per_token",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed0", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    seeds = [
        ("20260930", args.seed0 / "short_stop", args.seed0 / "equal_full_stop"),
        (
            "20261001",
            args.new / "short_stop_seed20261001",
            args.new / "equal_full_stop_seed20261001",
        ),
        (
            "20261002",
            args.new / "short_stop_seed20261002",
            args.new / "equal_full_stop_seed20261002",
        ),
    ]

    report: dict = {"seeds": {}, "across_seed": {}}
    method_metrics = {"short_stop": [], "equal_full_stop": []}
    for index, (seed, short_dir, full_dir) in enumerate(seeds):
        short_metrics = read_json(short_dir / "metrics.json")
        full_metrics = read_json(full_dir / "metrics.json")
        method_metrics["short_stop"].append(short_metrics)
        method_metrics["equal_full_stop"].append(full_metrics)
        short_rows = read_jsonl(short_dir / "eval_results.jsonl")
        full_rows = read_jsonl(full_dir / "eval_results.jsonl")
        report["seeds"][seed] = {
            "short_stop": short_metrics,
            "equal_full_stop": full_metrics,
            "truncation_rate_at_128": {
                "short_stop": truncation_rate(short_rows),
                "equal_full_stop": truncation_rate(full_rows),
            },
            "paired": paired_bootstrap(
                full_rows, short_rows, 20260930 + index, args.samples
            ),
        }

    for method, metrics_list in method_metrics.items():
        report["across_seed"][method] = {}
        for key in METRIC_KEYS:
            values = [float(metrics[key]) for metrics in metrics_list]
            report["across_seed"][method][key] = {
                "values": values,
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
            }

    deltas = {}
    for key in METRIC_KEYS:
        values = [
            float(full[key]) - float(short[key])
            for full, short in zip(
                method_metrics["equal_full_stop"], method_metrics["short_stop"]
            )
        ]
        deltas[key] = {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "positive_seeds": sum(value > 0 for value in values),
            "negative_seeds": sum(value < 0 for value in values),
        }
    report["across_seed"]["paired_delta_full_minus_short"] = deltas

    output = args.new / "multiseed_analysis.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["across_seed"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

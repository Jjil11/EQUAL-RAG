#!/usr/bin/env python3
"""Three-seed test analysis for frozen A-EQUAL-KL versus short_stop."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


METRICS = (
    "answer_recall", "citation_coverage", "grounded_edge_recall",
    "citation_validity", "mean_output_tokens", "grounded_utility_per_token",
)
SEEDS = ("20260930", "20261001", "20261002")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aequal", type=Path, required=True)
    ap.add_argument("--seed0-controls", type=Path, required=True)
    ap.add_argument("--other-controls", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()

    report: dict = {"protocol": {"split": "untouched test", "configuration": "frozen on dev"}, "seeds": {}}
    by_method = {"short_stop": [], "aequal_kl": []}
    for index, seed in enumerate(SEEDS):
        short_dir = (
            args.seed0_controls / "short_stop" if index == 0
            else args.other_controls / f"short_stop_seed{seed}"
        )
        aequal_dir = args.aequal / f"seed{seed}"
        short_metrics, aequal_metrics = load(short_dir / "metrics.json"), load(aequal_dir / "metrics.json")
        short_rows, aequal_rows = read_jsonl(short_dir / "eval_results.jsonl"), read_jsonl(aequal_dir / "eval_results.jsonl")
        by_method["short_stop"].append(short_metrics)
        by_method["aequal_kl"].append(aequal_metrics)
        report["seeds"][seed] = {
            "short_stop": short_metrics,
            "aequal_kl": aequal_metrics,
            "truncation_rate_at_128": {
                "short_stop": truncation_rate(short_rows),
                "aequal_kl": truncation_rate(aequal_rows),
            },
            "paired_aequal_minus_short": paired_bootstrap(
                aequal_rows, short_rows, 20261020 + index, args.samples
            ),
        }

    across = {}
    for method, rows in by_method.items():
        across[method] = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            across[method][metric] = {
                "values": values,
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
            }

    deltas = {}
    for metric in METRICS:
        values = [
            float(a[metric]) - float(s[metric])
            for a, s in zip(by_method["aequal_kl"], by_method["short_stop"])
        ]
        deltas[metric] = {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "improved_seeds": sum(v > 0 for v in values),
        }
    across["paired_delta_aequal_minus_short"] = deltas

    short_tokens = across["short_stop"]["mean_output_tokens"]["mean"]
    aequal_tokens = across["aequal_kl"]["mean_output_tokens"]["mean"]
    grounded_delta = deltas["grounded_edge_recall"]["mean"]
    utility_delta = deltas["grounded_utility_per_token"]["mean"]
    token_reduction = 1 - aequal_tokens / short_tokens
    report["across_seed"] = across
    report["pre_registered_gate"] = {
        "requirements": {
            "minimum_mean_token_reduction_fraction": 0.10,
            "minimum_mean_grounded_delta": -0.01,
            "utility_delta_must_be_positive": True,
        },
        "observed": {
            "mean_token_reduction_fraction": token_reduction,
            "mean_grounded_delta": grounded_delta,
            "mean_utility_delta": utility_delta,
        },
        "passes": token_reduction >= 0.10 and grounded_delta >= -0.01 and utility_delta > 0,
    }
    report["caution"] = (
        "The mean gate passes, but seed-level quality consistency must be reported; "
        "a negative seed cannot be hidden by the average."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"across_seed": across, "gate": report["pre_registered_gate"]}, indent=2))


if __name__ == "__main__":
    main()

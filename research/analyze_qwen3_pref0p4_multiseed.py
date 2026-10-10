#!/usr/bin/env python3
"""Three-seed Qwen3-14B analysis for the locked pref_alpha=0.4 method."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from analyze_qwen3_cached_aequal import utility_bootstrap
from analyze_second_pilot import paired_bootstrap, read_jsonl


METRICS = (
    "answer_recall",
    "citation_coverage",
    "grounded_edge_recall",
    "citation_validity",
    "mean_output_tokens",
    "grounded_utility_per_token",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hierarchical_bootstrap(seed_pairs: list[tuple[list[dict], list[dict]]], seed: int, samples: int) -> dict:
    """Resample both random-seed runs and shared evaluation questions."""
    count_fields = {
        "answer_recall": ("answers_found", "gold_edges"),
        "citation_coverage": ("answers_cited", "gold_edges"),
        "grounded_edge_recall": ("grounded_edges", "gold_edges"),
        "citation_validity": ("valid_citations", "citations"),
    }
    ids = sorted(str(row["id"]) for row in seed_pairs[0][0])
    aligned = []
    for method_rows, base_rows in seed_pairs:
        method_by_id = {str(row["id"]): row for row in method_rows}
        base_by_id = {str(row["id"]): row for row in base_rows}
        if sorted(method_by_id) != ids or sorted(base_by_id) != ids:
            raise ValueError("Evaluation IDs differ across seeds")
        aligned.append(([method_by_id[key] for key in ids], [base_by_id[key] for key in ids]))

    rng = random.Random(seed)
    metrics = list(count_fields) + ["output_tokens", "grounded_utility_per_token"]

    def aggregate(rows_by_seed: list[list[dict]], seed_indices: list[int], item_indices: list[int], metric: str) -> float:
        rows = [rows_by_seed[s][i] for s in seed_indices for i in item_indices]
        if metric == "output_tokens":
            return statistics.fmean(float(row["tokens"]) for row in rows)
        if metric == "grounded_utility_per_token":
            grounded = sum(float(row["grounded_edges"]) for row in rows)
            edges = sum(float(row["gold_edges"]) for row in rows)
            tokens = statistics.fmean(float(row["tokens"]) for row in rows)
            return (grounded / max(edges, 1.0)) / max(tokens, 1.0)
        numerator, denominator = count_fields[metric]
        return sum(float(row[numerator]) for row in rows) / max(
            1.0, sum(float(row[denominator]) for row in rows)
        )

    method_by_seed = [pair[0] for pair in aligned]
    base_by_seed = [pair[1] for pair in aligned]
    all_seeds = list(range(len(aligned)))
    all_items = list(range(len(ids)))
    result = {"seed_clusters": len(aligned), "question_clusters": len(ids), "bootstrap_samples": samples}
    for metric in metrics:
        mean_method = aggregate(method_by_seed, all_seeds, all_items, metric)
        mean_base = aggregate(base_by_seed, all_seeds, all_items, metric)
        deltas = []
        for _ in range(samples):
            chosen_seeds = [rng.randrange(len(aligned)) for _ in aligned]
            chosen_items = [rng.randrange(len(ids)) for _ in ids]
            deltas.append(
                aggregate(method_by_seed, chosen_seeds, chosen_items, metric)
                - aggregate(base_by_seed, chosen_seeds, chosen_items, metric)
            )
        deltas.sort()
        lo = deltas[int(0.025 * samples)]
        hi = deltas[min(samples - 1, int(0.975 * samples))]
        le_zero = sum(value <= 0 for value in deltas) / samples
        ge_zero = sum(value >= 0 for value in deltas) / samples
        result[metric] = {
            "method": mean_method,
            "baseline": mean_base,
            "delta": mean_method - mean_base,
            "ci95": [lo, hi],
            "p_two_sided": min(1.0, 2 * min(le_zero, ge_zero)),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    paths = {
        "20260930": (
            "runs/qwen3_14b_aequal/short_stop_seed20260930",
            "runs/qwen3_14b_compression_calibration/pref_0p4/test",
        ),
        "20260929": (
            "runs/qwen3_14b_pref0p4_multiseed/short_stop_seed20260929",
            "runs/qwen3_14b_pref0p4_multiseed/aequal_kl_pref0p4_seed20260929",
        ),
        "20261001": (
            "runs/qwen3_14b_pref0p4_multiseed/short_stop_seed20261001",
            "runs/qwen3_14b_pref0p4_multiseed/aequal_kl_pref0p4_seed20261001",
        ),
    }

    report: dict = {"seeds": {}, "aggregate": {}, "pooled_paired": {}}
    pooled_base, pooled_method = [], []
    seed_pairs = []
    for offset, (seed, (base_rel, method_rel)) in enumerate(paths.items()):
        base_dir, method_dir = args.root / base_rel, args.root / method_rel
        base_metrics = read_json(base_dir / "metrics.json")
        method_metrics = read_json(method_dir / "metrics.json")
        base_rows = read_jsonl(base_dir / "eval_results.jsonl")
        method_rows = read_jsonl(method_dir / "eval_results.jsonl")
        seed_pairs.append((method_rows, base_rows))
        for row in base_rows:
            pooled_base.append(row | {"id": f"{seed}:{row['id']}"})
        for row in method_rows:
            pooled_method.append(row | {"id": f"{seed}:{row['id']}"})
        deltas = {metric: method_metrics[metric] - base_metrics[metric] for metric in METRICS}
        deltas["relative_output_tokens"] = (
            method_metrics["mean_output_tokens"] / base_metrics["mean_output_tokens"] - 1
        )
        deltas["relative_utility"] = (
            method_metrics["grounded_utility_per_token"]
            / base_metrics["grounded_utility_per_token"] - 1
        )
        report["seeds"][seed] = {
            "baseline": base_metrics,
            "method": method_metrics,
            "delta_method_minus_baseline": deltas,
            "paired": paired_bootstrap(method_rows, base_rows, args.seed + offset, args.samples),
            "utility_paired": utility_bootstrap(
                method_rows, base_rows, args.seed + 100 + offset, args.samples
            ),
        }

    for metric in METRICS:
        base_values = [entry["baseline"][metric] for entry in report["seeds"].values()]
        method_values = [entry["method"][metric] for entry in report["seeds"].values()]
        deltas = [m - b for b, m in zip(base_values, method_values)]
        report["aggregate"][metric] = {
            "baseline_mean": statistics.fmean(base_values),
            "baseline_sd": statistics.stdev(base_values),
            "method_mean": statistics.fmean(method_values),
            "method_sd": statistics.stdev(method_values),
            "delta_mean": statistics.fmean(deltas),
            "delta_sd": statistics.stdev(deltas),
            "positive_seeds": sum(delta > 0 for delta in deltas),
            "negative_seeds": sum(delta < 0 for delta in deltas),
        }
    report["aggregate"]["relative_output_tokens"] = statistics.fmean(
        entry["delta_method_minus_baseline"]["relative_output_tokens"]
        for entry in report["seeds"].values()
    )
    report["aggregate"]["relative_utility"] = statistics.fmean(
        entry["delta_method_minus_baseline"]["relative_utility"]
        for entry in report["seeds"].values()
    )
    report["pooled_paired"] = paired_bootstrap(
        pooled_method, pooled_base, args.seed + 1000, args.samples
    )
    report["pooled_utility"] = utility_bootstrap(
        pooled_method, pooled_base, args.seed + 1001, args.samples
    )
    report["hierarchical_paired"] = hierarchical_bootstrap(
        seed_pairs, args.seed + 2000, args.samples
    )

    output = args.root / "qwen3_pref0p4_multiseed_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

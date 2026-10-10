#!/usr/bin/env python3
"""Paired analysis for the corrected 2048-context EQUAL-RAG pilot."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


METRICS = (
    "answer_recall",
    "citation_coverage",
    "grounded_edge_recall",
    "citation_validity",
    "output_tokens",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align(a: list[dict], b: list[dict]) -> tuple[list[dict], list[dict]]:
    a_by_id = {str(row["id"]): row for row in a}
    b_by_id = {str(row["id"]): row for row in b}
    if set(a_by_id) != set(b_by_id):
        raise ValueError("Evaluation IDs differ between methods")
    ids = sorted(a_by_id)
    return [a_by_id[i] for i in ids], [b_by_id[i] for i in ids]


def paired_bootstrap(a: list[dict], b: list[dict], seed: int, samples: int) -> dict:
    a, b = align(a, b)
    rng = random.Random(seed)
    n = len(a)
    result = {"n": n, "bootstrap_samples": samples}

    count_fields = {
        "answer_recall": ("answers_found", "gold_edges"),
        "citation_coverage": ("answers_cited", "gold_edges"),
        "grounded_edge_recall": ("grounded_edges", "gold_edges"),
        "citation_validity": ("valid_citations", "citations"),
    }

    def aggregate(rows: list[dict], metric: str, indices: list[int] | None = None) -> float:
        chosen = rows if indices is None else [rows[i] for i in indices]
        if metric == "output_tokens":
            return statistics.fmean(float(row["tokens"]) for row in chosen)
        numerator, denominator = count_fields[metric]
        num = sum(float(row[numerator]) for row in chosen)
        den = sum(float(row[denominator]) for row in chosen)
        return num / max(1.0, den)

    for metric in METRICS:
        mean_a = aggregate(a, metric)
        mean_b = aggregate(b, metric)
        boot = []
        for _ in range(samples):
            indices = [rng.randrange(n) for _ in range(n)]
            boot.append(aggregate(a, metric, indices) - aggregate(b, metric, indices))
        boot.sort()
        lo = boot[int(0.025 * samples)]
        hi = boot[min(samples - 1, int(0.975 * samples))]
        le_zero = sum(value <= 0 for value in boot) / samples
        ge_zero = sum(value >= 0 for value in boot) / samples
        p_two = min(1.0, 2 * min(le_zero, ge_zero))
        result[metric] = {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta_a_minus_b": mean_a - mean_b,
            "ci95": [lo, hi],
            "p_two_sided": p_two,
        }
    return result


def truncation_rate(rows: list[dict], limit: int = 128) -> float:
    return sum(float(row.get("output_tokens", row.get("tokens", 0))) >= limit for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    methods = ("short_stop", "equal_cov", "equal_cov_stop", "equal_full_stop")
    rows = {m: read_jsonl(args.second / m / "eval_results.jsonl") for m in methods}
    short_sft = read_jsonl(args.first / "short_sft" / "eval_results.jsonl")

    comparisons = {
        "equal_full_stop_vs_short_stop": paired_bootstrap(
            rows["equal_full_stop"], rows["short_stop"], args.seed, args.samples
        ),
        "equal_full_stop_vs_equal_cov_stop": paired_bootstrap(
            rows["equal_full_stop"], rows["equal_cov_stop"], args.seed + 1, args.samples
        ),
        "short_stop_vs_first_short_sft": paired_bootstrap(
            rows["short_stop"], short_sft, args.seed + 2, args.samples
        ),
    }
    report = {
        "truncation_rate_at_128": {m: truncation_rate(v) for m, v in rows.items()},
        "comparisons": comparisons,
    }
    out = args.second / "paired_analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

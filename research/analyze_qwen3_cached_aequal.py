#!/usr/bin/env python3
"""Paired dev-set analysis for cached-anchor Qwen3-14B A-EQUAL-KL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from analyze_second_pilot import paired_bootstrap, read_jsonl, truncation_rate


def utility_bootstrap(a: list[dict], b: list[dict], seed: int, samples: int) -> dict:
    a_by_id = {str(row["id"]): row for row in a}
    b_by_id = {str(row["id"]): row for row in b}
    if set(a_by_id) != set(b_by_id):
        raise ValueError("Evaluation IDs differ between methods")
    ids = sorted(a_by_id)
    a = [a_by_id[key] for key in ids]
    b = [b_by_id[key] for key in ids]
    rng = random.Random(seed)

    def utility(rows: list[dict], indices: list[int]) -> float:
        grounded = sum(rows[i]["grounded_edges"] for i in indices)
        edges = sum(rows[i]["gold_edges"] for i in indices)
        tokens = sum(rows[i]["tokens"] for i in indices) / len(indices)
        return (grounded / max(edges, 1)) / max(tokens, 1)

    full = list(range(len(a)))
    mean_a, mean_b = utility(a, full), utility(b, full)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(a)) for _ in a]
        deltas.append(utility(a, indices) - utility(b, indices))
    deltas.sort()
    lo = deltas[int(0.025 * samples)]
    hi = deltas[min(samples - 1, int(0.975 * samples))]
    le_zero = sum(x <= 0 for x in deltas) / samples
    ge_zero = sum(x >= 0 for x in deltas) / samples
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta_a_minus_b": mean_a - mean_b,
        "relative_delta": (mean_a / mean_b) - 1,
        "ci95": [lo, hi],
        "p_two_sided": min(1.0, 2 * min(le_zero, ge_zero)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    baseline = read_jsonl(
        args.root / "runs/qwen3_14b_aequal_conservative/short_stop_dev/eval_results.jsonl"
    )
    method = read_jsonl(
        args.root / "runs/qwen3_14b_aequal_cached/aequal_kl_seed20260930/dev/eval_results.jsonl"
    )
    report = {
        "comparison": "aequal_kl_minus_short_stop",
        "paired": paired_bootstrap(method, baseline, args.seed, args.samples),
        "grounded_utility_per_token": utility_bootstrap(
            method, baseline, args.seed + 1, args.samples
        ),
        "truncation_rate_128": {
            "aequal_kl": truncation_rate(method, 128),
            "short_stop": truncation_rate(baseline, 128),
        },
    }
    output = args.root / "paired_dev_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

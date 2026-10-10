#!/usr/bin/env python3
"""Paired bootstrap analysis for the first EQUAL-RAG feasibility pilot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


METHODS = ["long_sft", "short_sft", "length_pref", "equal_rag"]


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {row["id"]: row for row in map(json.loads, f)}


def q(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(int(p * len(values)), len(values) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    root = Path(args.root)
    data = {m: load(root / m / "eval_results.jsonl") for m in METHODS}
    ids = sorted(set.intersection(*(set(x) for x in data.values())))
    rng = random.Random(args.seed)
    comparisons = {}
    for method in ["long_sft", "length_pref", "equal_rag"]:
        edge_deltas, token_deltas = [], []
        for _ in range(args.samples):
            sample = [ids[rng.randrange(len(ids))] for _ in ids]
            gold = sum(data[method][i]["gold_edges"] for i in sample)
            m_edges = sum(data[method][i]["grounded_edges"] for i in sample) / gold
            s_edges = sum(data["short_sft"][i]["grounded_edges"] for i in sample) / gold
            edge_deltas.append(m_edges - s_edges)
            token_deltas.append(
                sum(data[method][i]["tokens"] - data["short_sft"][i]["tokens"] for i in sample) / len(sample)
            )
        comparisons[f"{method}_vs_short_sft"] = {
            "grounded_edge_delta_ci95": [q(edge_deltas, 0.025), q(edge_deltas, 0.975)],
            "grounded_edge_delta_p_two_sided": 2 * min(
                sum(x <= 0 for x in edge_deltas) / len(edge_deltas),
                sum(x >= 0 for x in edge_deltas) / len(edge_deltas),
            ),
            "token_delta_ci95": [q(token_deltas, 0.025), q(token_deltas, 0.975)],
        }
    diagnostics = {
        m: {
            "truncation_rate_at_128": sum(x["tokens"] >= 128 for x in data[m].values()) / len(ids),
            "n": len(ids),
        }
        for m in METHODS
    }
    result = {"comparisons": comparisons, "diagnostics": diagnostics, "bootstrap_samples": args.samples}
    (root / "paired_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

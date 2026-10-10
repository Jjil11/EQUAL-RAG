#!/usr/bin/env python3
"""Summarize the four-arm EQUAL-RAG feasibility pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ORDER = ["long_sft", "short_sft", "length_pref", "equal_rag"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.runs)
    rows = {}
    for method in ORDER:
        path = root / method / "metrics.json"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        rows[method] = json.loads(path.read_text(encoding="utf-8"))
    short = rows["short_sft"]
    for method, metrics in rows.items():
        metrics["delta_tokens_vs_short"] = metrics["mean_output_tokens"] - short["mean_output_tokens"]
        metrics["delta_grounded_vs_short"] = metrics["grounded_edge_recall"] - short["grounded_edge_recall"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# EQUAL-RAG first feasibility pilot",
        "",
        "This is a disjoint feasibility split drawn from the ALCE evaluation collection; it is not an official benchmark result.",
        "",
        "| Method | Answer recall | Citation coverage | Grounded-edge recall | Valid citations | Mean output tokens | Grounded utility/token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ORDER:
        m = rows[method]
        lines.append(
            f"| {method} | {m['answer_recall']:.4f} | {m['citation_coverage']:.4f} | "
            f"{m['grounded_edge_recall']:.4f} | {m['citation_validity']:.4f} | "
            f"{m['mean_output_tokens']:.2f} | {m['grounded_utility_per_token']:.6f} |"
        )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

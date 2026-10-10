#!/usr/bin/env python3
"""Apply the pre-registered C-EQUAL development-set selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    runs = args.root / "runs"
    short = load(runs / "cequal_dev_controls" / "short_stop" / "metrics.json")
    fixed = load(runs / "cequal_dev_controls" / "equal_full_stop" / "metrics.json")
    threshold = short["grounded_edge_recall"] - args.tolerance

    candidates = {}
    for path in sorted((runs / "cequal_dev_calibration").glob("c*/metrics.json")):
        metrics = load(path)
        metrics["grounded_delta_vs_short"] = (
            metrics["grounded_edge_recall"] - short["grounded_edge_recall"]
        )
        metrics["token_delta_vs_short"] = (
            metrics["mean_output_tokens"] - short["mean_output_tokens"]
        )
        metrics["token_reduction_fraction_vs_short"] = (
            1 - metrics["mean_output_tokens"] / short["mean_output_tokens"]
        )
        metrics["passes_quality_gate"] = metrics["grounded_edge_recall"] >= threshold
        candidates[path.parent.name] = metrics

    survivors = [
        (name, metrics) for name, metrics in candidates.items()
        if metrics["passes_quality_gate"]
    ]
    survivors.sort(
        key=lambda item: (
            item[1]["mean_output_tokens"],
            -item[1]["grounded_utility_per_token"],
        )
    )
    selected = survivors[0][0] if survivors else None
    report = {
        "selection_protocol": {
            "split": "development only",
            "quality_reference": "short_stop",
            "quality_tolerance": args.tolerance,
            "minimum_grounded_edge_recall": threshold,
            "ranking_after_gate": ["minimum mean_output_tokens", "maximum grounded_utility_per_token"],
        },
        "controls": {"short_stop": short, "equal_full_stop": fixed},
        "candidates": candidates,
        "selected": selected,
        "decision": (
            "No C-EQUAL candidate passed the pre-registered quality gate."
            if selected is None else f"Selected {selected} without inspecting test results."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

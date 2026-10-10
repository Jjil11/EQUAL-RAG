#!/usr/bin/env python3
"""Measure exact repeated comma-separated answer units in QAMPARI outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize(unit: str) -> str:
    return re.sub(r"\s+", " ", unit.strip().lower().rstrip(". ;"))


def summarize(path: Path, cap: int, unit_mode: str) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    per_row = []
    for row in rows:
        raw_parts = row["output"].splitlines() if unit_mode == "line" else row["output"].split(",")
        parts = [normalize(x) for x in raw_parts]
        parts = [x for x in parts if x]
        excess = len(parts) - len(set(parts))
        per_row.append(
            {
                "id": row["id"],
                "tokens": row["tokens"],
                "hit_cap": row["tokens"] >= cap,
                "units": len(parts),
                "duplicate_units": excess,
                "duplicate_fraction": excess / max(1, len(parts)),
            }
        )
    capped = [r for r in per_row if r["hit_cap"]]

    def aggregate(items: list[dict]) -> dict:
        n = len(items)
        return {
            "n": n,
            "any_exact_duplicate_rate": sum(r["duplicate_units"] > 0 for r in items) / max(1, n),
            "mean_duplicate_units": sum(r["duplicate_units"] for r in items) / max(1, n),
            "mean_duplicate_fraction": sum(r["duplicate_fraction"] for r in items) / max(1, n),
        }

    return {
        "path": str(path),
        "cap": cap,
        "unit_mode": unit_mode,
        "hit_cap_rate": len(capped) / max(1, len(per_row)),
        "all_outputs": aggregate(per_row),
        "cap_hit_outputs": aggregate(capped),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", nargs=3, metavar=("LABEL", "PATH", "CAP"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--unit-mode", choices=["comma", "line"], default="comma")
    args = ap.parse_args()
    report = {label: summarize(Path(path), int(cap), args.unit_mode) for label, path, cap in args.input}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

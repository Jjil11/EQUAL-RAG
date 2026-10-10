#!/usr/bin/env python3
"""Merge trajectory shards into training rows and emit mining diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.train)
    trajectories = {}
    raw = []
    for shard in args.shards:
        for item in read_jsonl(shard):
            raw.append(item)
            if item.get("has_trajectory"):
                trajectories[item["id"]] = item
    if len(raw) != len(rows):
        raise ValueError(f"Expected {len(rows)} mined rows, found {len(raw)}")
    if len({item["id"] for item in raw}) != len(rows):
        raise ValueError("Trajectory shards contain missing or duplicate ids")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            merged = dict(row)
            trajectory = trajectories.get(row["id"])
            if trajectory:
                for key in (
                    "trajectory_prefix", "trajectory_continuation",
                    "trajectory_repeat_unit", "trajectory_first_index",
                    "trajectory_source_index", "tokens", "cap_hit",
                ):
                    merged[key] = trajectory[key]
                merged["has_trajectory"] = True
            else:
                merged["has_trajectory"] = False
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")

    mined_items = list(trajectories.values())
    summary = {
        "n": len(rows),
        "mined": len(mined_items),
        "mined_rate": len(mined_items) / len(rows),
        "cap_hits": sum(bool(item["cap_hit"]) for item in raw),
        "cap_hit_rate": sum(bool(item["cap_hit"]) for item in raw) / len(rows),
        "mined_cap_hits": sum(bool(item["cap_hit"]) for item in mined_items),
        "mean_first_repeat_index": (
            sum(item["trajectory_first_index"] for item in mined_items) / len(mined_items)
            if mined_items else None
        ),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

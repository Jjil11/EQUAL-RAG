#!/usr/bin/env python3
"""Print structural statistics for MultiSpanQA without content examples."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def shape(value):
    if isinstance(value, dict):
        return {key: type(item).__name__ for key, item in value.items()}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "first_type": type(value[0]).__name__ if value else None}
    return {"type": type(value).__name__}


def audit(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    report = {"file": path.name, "top": shape(value)}
    rows = value.get("data", value) if isinstance(value, dict) else value
    if isinstance(rows, list):
        report["rows"] = len(rows)
        if rows:
            report["row_shape"] = shape(rows[0])
            for key, item in rows[0].items():
                if isinstance(item, list) and item:
                    report[f"{key}_item_shape"] = shape(item[0])
        if rows and "label" in rows[0]:
            report["label_vocabulary"] = sorted({label for row in rows for label in row["label"]})
            report["label_context_length_match"] = sum(len(row["label"]) == len(row["context"]) for row in rows)
            report["context_length"] = {
                "min": min(len(row["context"]) for row in rows),
                "max": max(len(row["context"]) for row in rows),
                "mean": sum(len(row["context"]) for row in rows) / len(rows),
            }
        if rows and "num_span" in rows[0]:
            counts = Counter(row["num_span"] for row in rows)
            report["num_span_count_distribution"] = dict(sorted(counts.items()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    result = {split: audit(args.data_dir / f"{split}.json") for split in ("train", "valid", "test")}
    output = args.data_dir / "structure_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit official FanOutQA dev answers before constructing EQUAL-RAG splits."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def leaves(nodes):
    out = []
    for node in nodes:
        if node.get("decomposition"):
            out.extend(leaves(node["decomposition"]))
        else:
            out.append(node)
    return out


def units(answer):
    if isinstance(answer, dict):
        return [f"{key}: {value}" for key, value in answer.items()]
    if isinstance(answer, list):
        return [str(value) for value in answer]
    return [str(answer)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    types = collections.Counter(type(row["answer"]).__name__ for row in rows)
    unit_counts = [len(units(row["answer"])) for row in rows]
    leaf_counts = [len(leaves(row["decomposition"])) for row in rows]
    eligible = [row for row in rows if isinstance(row["answer"], (dict, list)) and len(units(row["answer"])) >= 3]
    report = {
        "n": len(rows), "answer_types": dict(types),
        "unit_count": {"mean": statistics.fmean(unit_counts), "median": statistics.median(unit_counts),
                       "max": max(unit_counts), "ge3": sum(x >= 3 for x in unit_counts),
                       "ge5": sum(x >= 5 for x in unit_counts), "ge10": sum(x >= 10 for x in unit_counts)},
        "leaf_evidence_count": {"mean": statistics.fmean(leaf_counts), "median": statistics.median(leaf_counts),
                                "max": max(leaf_counts)},
        "eligible_multiunit_n": len(eligible),
        "eligible_type_counts": dict(collections.Counter(type(row["answer"]).__name__ for row in eligible)),
        "examples": [{"id": row["id"], "question": row["question"], "units": units(row["answer"]),
                      "leaf_count": len(leaves(row["decomposition"]))} for row in eligible[:5]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

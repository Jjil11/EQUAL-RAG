#!/usr/bin/env python3
"""Integrity audit for converted EQUAL-RAG JSONL datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    ids, report = {}, {}
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in (args.data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        split_ids = [str(row["id"]) for row in rows]
        ids[split] = set(split_ids)
        missing_alias = 0
        malformed = 0
        duplicate_gold = 0
        for row in rows:
            required = {"id", "prompt", "docs", "edges", "short", "long", "negative_drop", "negative_swap", "negative_redundant"}
            malformed += not required.issubset(row)
            doc = norm(" ".join(item["text"] for item in row["docs"]))
            answers = [norm(edge["answer"]) for edge in row["edges"]]
            duplicate_gold += len(answers) != len(set(answers))
            missing_alias += sum(not any(norm(alias) in doc for alias in edge["aliases"]) for edge in row["edges"])
        report[split] = {
            "n": len(rows), "unique_ids": len(set(split_ids)), "malformed_rows": malformed,
            "edges_with_no_alias_in_evidence": missing_alias, "rows_with_duplicate_gold": duplicate_gold,
        }
    report["cross_split_overlap"] = {
        "train_dev": len(ids["train"] & ids["dev"]), "train_test": len(ids["train"] & ids["test"]),
        "dev_test": len(ids["dev"] & ids["test"]),
    }
    failures = sum(item.get("malformed_rows", 0) + item.get("edges_with_no_alias_in_evidence", 0) + item.get("rows_with_duplicate_gold", 0) for item in report.values())
    failures += sum(report["cross_split_overlap"].values())
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

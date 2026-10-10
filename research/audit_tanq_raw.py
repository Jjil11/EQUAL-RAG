#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def q(values, p):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    key_counts = collections.Counter()
    n_answers, n_pass, n_columns, n_cells, proof_counts = [], [], [], [], []
    table_types = collections.Counter()
    example_keys = []
    for row in rows:
        key_counts.update(row.keys())
        answers = row.get("answer_list") or []
        passed = [a for a in answers if a.get("filter_pass", True)]
        n_answers.append(len(answers)); n_pass.append(len(passed))
        cols = set()
        cells = 0
        proofs = 0
        for answer in passed:
            init_proofs = answer.get("init_answer_proof") or []
            proofs += len(init_proofs)
            for ext in answer.get("extension_answer") or []:
                cols.add(str(ext.get("extension_property_label", ext.get("extension_property_id", ""))))
                cells += 1
                proofs += len(ext.get("proof") or [])
        n_columns.append(len(cols)); n_cells.append(cells); proof_counts.append(proofs)
        table_types[type(row.get("answer_table")).__name__] += 1
        if len(example_keys) < 3:
            first_answer = answers[0] if answers else {}
            first_ext = (first_answer.get("extension_answer") or [{}])[0]
            first_proof = (first_ext.get("proof") or [{}])[0]
            example_keys.append({
                "qid": row.get("init_qid"),
                "keys": sorted(row),
                "answer_keys": sorted(answers[0]) if answers else [],
                "extension_keys": sorted((answers[0].get("extension_answer") or [{}])[0]) if answers and answers[0].get("extension_answer") else [],
                "table_type": type(row.get("answer_table")).__name__,
                "question": row.get("ext_question_rephrased"),
                "answer_composed": first_answer.get("init_answer_composed"),
                "extension_property": first_ext.get("extension_property_label"),
                "extension_entity": first_ext.get("extension_entity"),
                "proof_keys": sorted(first_proof),
                "proof_sample": first_proof,
            })
    def stats(xs):
        return {"mean": statistics.mean(xs) if xs else 0, "median": statistics.median(xs) if xs else 0,
                "p90": q(xs, .9), "p95": q(xs, .95), "max": max(xs) if xs else 0,
                "ge_5": sum(x >= 5 for x in xs), "ge_10": sum(x >= 10 for x in xs)}
    report = {
        "n": len(rows), "top_level_key_counts": key_counts,
        "answer_table_types": table_types,
        "answers": stats(n_answers), "passed_rows": stats(n_pass),
        "columns": stats(n_columns), "extension_cells": stats(n_cells),
        "proofs": stats(proof_counts), "examples": example_keys,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

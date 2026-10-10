#!/usr/bin/env python3
"""Build a controlled TANQ oracle-generation split for EQUAL-RAG.

Each table row is one independently citable evidence unit.  The official TANQ
test set is not read by this builder; the official dev collection is partitioned
by a hash of qid into train, development-screen, and local holdout subsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def clean(value: str) -> str:
    return " ".join(str(value).replace("\ufffd", " ").split())


def parse_table(table: str):
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    if len(lines) < 2:
        return [], []
    header = [clean(x) for x in lines[0].split("|")]
    rows = [[clean(x) for x in line.split("|")] for line in lines[1:]]
    rows = [row for row in rows if len(row) == len(header) and all(row)]
    return header, rows


def render_row(header, values):
    entity = values[0]
    attrs = " | ".join(f"{header[i]}={values[i]}" for i in range(1, len(header)))
    return f"{entity} | {attrs}" if attrs else entity


def make_example(raw):
    header, table_rows = parse_table(raw.get("answer_table", ""))
    if not (3 <= len(table_rows) <= 12 and 2 <= len(header) <= 8):
        return None
    docs, edges, targets = [], [], []
    for index, values in enumerate(table_rows, start=1):
        canonical = render_row(header, values)
        docs.append({"id": index, "title": values[0], "text": canonical})
        edges.append({
            "answer_idx": index - 1, "answer": canonical,
            "aliases": [canonical], "doc_id": index,
        })
        targets.append(f"{canonical} [{index}]")
    evidence = "\n".join(f"[{doc['id']}] {doc['text']}" for doc in docs)
    question = clean(raw.get("ext_question_rephrased") or raw.get("ext_question_cleaned") or raw.get("ext_question"))
    prompt = (
        "Construct the complete answer table using only the evidence rows below. "
        "Output exactly one table row per line, preserve the shown column names and values, "
        "and put the supporting evidence number at the end of each row. Do not repeat rows.\n\n"
        f"{evidence}\n\nQuestion: {question}\nAnswer:"
    )
    short = "\n".join(targets)
    source_qid = str(raw["init_qid"])
    instance_fingerprint = hashlib.sha256(
        (source_qid + "\n" + question + "\n" + short).encode()
    ).hexdigest()[:16]
    instance_id = f"{source_qid}__{instance_fingerprint}"
    rng = random.Random(int(hashlib.sha256(instance_id.encode()).hexdigest()[:16], 16))
    drop = rng.randrange(len(targets))
    swapped = list(targets)
    swap = rng.randrange(len(swapped))
    wrong_doc = 1 + (swap + 1) % len(swapped)
    swapped[swap] = swapped[swap].rsplit("[", 1)[0] + f"[{wrong_doc}]"
    return {
        "id": instance_id, "source_qid": source_qid,
        "question": question, "prompt": prompt,
        "docs": docs, "edges": edges, "short": short,
        "long": "\n".join(f"The evidence-supported table row is {x}." for x in targets),
        "negative_drop": "\n".join(x for i, x in enumerate(targets) if i != drop),
        "negative_swap": "\n".join(swapped),
        "negative_redundant": short + "\n" + targets[-1],
        "tanq_header": header, "tanq_rows": len(table_rows),
        "source_split": "official_dev",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mistral-model", required=True)
    ap.add_argument("--qwen-model", required=True)
    ap.add_argument("--train", type=int, default=250)
    ap.add_argument("--dev", type=int, default=80)
    ap.add_argument("--holdout", type=int, default=80)
    ap.add_argument("--max-target-tokens", type=int, default=240)
    ap.add_argument("--max-total-tokens", type=int, default=2048)
    args = ap.parse_args()
    mtok = AutoTokenizer.from_pretrained(args.mistral_model, use_fast=True)
    qtok = AutoTokenizer.from_pretrained(args.qwen_model, use_fast=True)
    raw_rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    eligible, rejected = [], {"structure": 0, "length": 0, "exact_duplicate": 0}
    for raw in raw_rows:
        row = make_example(raw)
        if row is None:
            rejected["structure"] += 1; continue
        lengths = {}
        valid = True
        for name, tok in (("mistral", mtok), ("qwen", qtok)):
            target = len(tok(row["short"], add_special_tokens=False).input_ids)
            prompt = len(tok(row["prompt"], add_special_tokens=False).input_ids)
            lengths[name] = {"prompt": prompt, "target": target, "total": prompt + target}
            valid &= target <= args.max_target_tokens and prompt + target <= args.max_total_tokens
        if not valid:
            rejected["length"] += 1; continue
        row["token_lengths"] = lengths
        eligible.append(row)
    deduplicated = {}
    for row in eligible:
        if row["id"] in deduplicated:
            rejected["exact_duplicate"] += 1
        else:
            deduplicated[row["id"]] = row
    eligible = list(deduplicated.values())
    # TANQ can contain several extension questions derived from the same initial
    # question.  Keep every source_qid in exactly one split, while retaining a
    # unique instance id for paired statistics.
    groups = {}
    for row in eligible:
        groups.setdefault(row["source_qid"], []).append(row)
    ordered_groups = sorted(
        groups.values(),
        key=lambda rows: (
            -len(rows),
            hashlib.sha256(("20261006:" + rows[0]["source_qid"]).encode()).hexdigest(),
        ),
    )
    needed = args.train + args.dev + args.holdout
    if len(eligible) < needed:
        raise SystemExit(f"eligible={len(eligible)} < requested={needed}; rejected={rejected}")
    if len(eligible) != needed:
        # Selection remains group-safe if a future invocation requests fewer
        # instances: take whole groups until the requested budget is reached.
        chosen_groups, count = [], 0
        for group in ordered_groups:
            if count >= needed:
                break
            chosen_groups.append(group); count += len(group)
        ordered_groups = chosen_groups
    targets = {"train": args.train, "dev": args.dev, "holdout": args.holdout}
    splits = {name: [] for name in targets}
    for group in ordered_groups:
        # Largest normalized deficit first.  This deterministic grouped
        # allocation closely matches requested sizes without source leakage.
        destination = max(
            targets,
            key=lambda name: (targets[name] - len(splits[name])) / targets[name],
        )
        splits[destination].extend(group)
    all_ids = [row["id"] for rows in splits.values() for row in rows]
    assert len(all_ids) == len(set(all_ids)), "instance IDs must be unique"
    qid_sets = {name: {row["source_qid"] for row in rows} for name, rows in splits.items()}
    assert not (qid_sets["train"] & qid_sets["dev"])
    assert not (qid_sets["train"] & qid_sets["holdout"])
    assert not (qid_sets["dev"] & qid_sets["holdout"])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(args.input), "source_n": len(raw_rows), "eligible_n": len(eligible),
        "rejected": rejected,
        "selection": "source_qid-grouped deterministic deficit allocation; unique content-hash instance IDs",
        "warning": "Controlled oracle-generator split from official TANQ dev; not an official benchmark score.",
        "official_test_status": "unread and not downloaded",
        "splits": {name: {
            "n": len(rows),
            "mean_rows": sum(r["tanq_rows"] for r in rows) / len(rows),
            "mean_edges": sum(len(r["edges"]) for r in rows) / len(rows),
            "max_rows": max(r["tanq_rows"] for r in rows),
            "unique_source_qids": len({r["source_qid"] for r in rows}),
            "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(),
        } for name, rows in splits.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

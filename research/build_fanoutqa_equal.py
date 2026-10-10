#!/usr/bin/env python3
"""Build frozen evidence-unit EQUAL-RAG splits from official FanOutQA dev data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


def norm(value) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def flatten(value):
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


def leaves(nodes):
    out = []
    for node in nodes:
        if node.get("decomposition"):
            out.extend(leaves(node["decomposition"]))
        else:
            out.append(node)
    return out


def answer_units(answer):
    if isinstance(answer, dict):
        return [(str(key), str(value), f"{key}: {value}") for key, value in answer.items()]
    if isinstance(answer, list):
        return [(str(value), str(value), str(value)) for value in answer]
    return [(str(answer), str(answer), str(answer))]


def choose_doc(key: str, value: str, leaf_rows: list[dict]) -> int:
    nk, nv = norm(key), norm(value)
    scored = []
    for index, leaf in enumerate(leaf_rows, 1):
        question, answer = norm(leaf.get("question", "")), norm(flatten(leaf.get("answer", "")))
        score = 0
        if nv and (nv in answer or answer in nv): score += 4
        if nk and nk in question: score += 3
        if nk and nk in answer: score += 1
        scored.append((score, index))
    score, index = max(scored)
    return index if score else 1


def convert(row: dict) -> dict:
    leaf_rows = leaves(row["decomposition"])
    docs = []
    for index, leaf in enumerate(leaf_rows, 1):
        evidence = leaf.get("evidence") or {}
        title = evidence.get("title", "")
        text = f"Question: {leaf.get('question', '')}\nVerified answer: {flatten(leaf.get('answer', ''))}."
        docs.append({"id": index, "title": title, "text": text})
    deduplicated, seen_units = [], set()
    for item in answer_units(row["answer"]):
        marker = norm(item[2])
        if marker not in seen_units:
            deduplicated.append(item)
            seen_units.add(marker)
    edges = []
    for index, (key, value, full) in enumerate(deduplicated):
        edges.append({"answer_idx": index, "answer": full, "aliases": [full],
                      "doc_id": choose_doc(key, value, leaf_rows)})
    # The compact evidence-provided setting uses the official decomposition
    # answers as oracle evidence. Make the exact evaluated key-value unit
    # explicit in its mapped evidence block rather than relying on fuzzy joins.
    for edge in edges:
        docs[edge["doc_id"] - 1]["text"] += f"\nSupported final answer unit: {edge['answer']}."
    short = ", ".join(f"{edge['answer']} [{edge['doc_id']}]" for edge in edges) + "."
    long = " ".join(f"The evidence supports {edge['answer']} [{edge['doc_id']}]." for edge in edges)
    negative_drop = ", ".join(
        f"{edge['answer']} [{edge['doc_id']}]" for edge in edges[:-1]
    ) + "."
    negative_redundant = short + " To repeat the complete answer: " + short
    evidence_block = "\n".join(f"[{doc['id']}] {doc['title']}: {doc['text']}" for doc in docs)
    prompt = (
        "Answer the fan-out question using only the verified evidence units. Give a concise comma-separated "
        "list of key-value answers, cite the supporting unit after each answer, include every supported item, "
        "and do not repeat items.\n\n" + evidence_block + f"\n\nQuestion: {row['question']}\nAnswer:"
    )
    return {"id": f"fanoutqa-{row['id']}", "question": row["question"], "prompt": prompt,
            "docs": docs, "edges": edges, "short": short, "long": long,
            "negative_drop": negative_drop, "negative_swap": negative_drop,
            "negative_redundant": negative_redundant, "source_categories": row.get("categories", [])}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--train", type=int, default=180)
    p.add_argument("--dev", type=int, default=50)
    p.add_argument("--test", type=int, default=50)
    p.add_argument("--tokenizer", action="append", default=[])
    p.add_argument("--target-token-cap", type=int, default=240)
    args = p.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    eligible = [row for row in raw if isinstance(row["answer"], (dict, list)) and len(answer_units(row["answer"])) >= 3]
    eligible.sort(key=lambda row: hashlib.sha256(f"{args.seed}:{row['id']}".encode()).digest())
    tokenizers = [AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True) for path in args.tokenizer]
    converted_pool, excluded_for_length = [], []
    for row in eligible:
        converted = convert(row)
        lengths = [len(tok(converted["short"], add_special_tokens=False)["input_ids"]) for tok in tokenizers]
        if lengths and max(lengths) > args.target_token_cap:
            excluded_for_length.append({"id": converted["id"], "target_lengths": lengths})
        else:
            converted_pool.append(converted)
    if args.train + args.dev + args.test > len(converted_pool):
        raise ValueError("Requested split sizes exceed eligible examples")
    split_rows = {"train": converted_pool[:args.train], "dev": converted_pool[args.train:args.train+args.dev],
                  "test": converted_pool[args.train+args.dev:args.train+args.dev+args.test]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "FanOutQA official dev controlled transfer", "source_sha256": sha(args.input),
                "seed": args.seed, "selection": "eligible dict/list answers with >=3 units; SHA256 order; target length filter",
                "target_token_cap": args.target_token_cap, "tokenizers": args.tokenizer,
                "excluded_for_length": excluded_for_length, "splits": {}}
    seen = set()
    for split, rows in split_rows.items():
        converted = rows
        ids = {row["id"] for row in converted}
        if seen & ids: raise ValueError("Cross-split overlap")
        seen |= ids
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted), encoding="utf-8")
        manifest["splits"][split] = {"n": len(converted),
            "mean_edges": sum(len(row["edges"]) for row in converted) / len(converted),
            "max_edges": max(len(row["edges"]) for row in converted), "sha256": sha(path)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert official MultiSpanQA BIO data into frozen EQUAL-RAG splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    for suffix in ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m"):
        text = text.replace(" " + suffix, suffix)
    return " ".join(text.split())


def extract_spans(tokens: list[str], labels: list[str]) -> list[str]:
    if len(tokens) != len(labels):
        raise ValueError("BIO labels and context lengths differ")
    spans, current = [], []
    for token, label in zip(tokens, labels):
        if label == "B" or (label == "I" and not current):
            if current:
                spans.append(detokenize(current))
            current = [token]
        elif label == "I":
            current.append(token)
        elif current:
            spans.append(detokenize(current))
            current = []
    if current:
        spans.append(detokenize(current))
    return spans


def short_target(edges: list[dict]) -> str:
    return ", ".join(f"{edge['answer']} [1]" for edge in edges) + "."


def convert(raw: dict, rng: random.Random) -> dict:
    question = detokenize(raw["question"])
    context = detokenize(raw["context"])
    extracted = extract_spans(raw["context"], raw["label"])
    if len(extracted) != raw["num_span"]:
        raise ValueError(f"{raw['id']}: extracted {len(extracted)} spans, expected {raw['num_span']}")
    # MultiSpanQA labels span positions. Generative list QA evaluates answer
    # identities, so collapse the rare case where the same answer text is
    # annotated at multiple positions in one passage.
    answers, seen = [], set()
    for answer in extracted:
        key = " ".join(re.sub(r"[^a-z0-9 ]+", " ", answer.lower()).split())
        if key not in seen:
            answers.append(answer)
            seen.add(key)
    edges = [
        {"answer_idx": index, "answer": answer, "aliases": [answer], "doc_id": 1}
        for index, answer in enumerate(answers)
    ]
    concise = short_target(edges)
    long = " ".join(f"The passage supports {edge['answer']} as an answer [1]." for edge in edges)
    drop = rng.randrange(len(edges))
    negative_drop = short_target([edge for index, edge in enumerate(edges) if index != drop])
    negative_redundant = concise + " To repeat the supported answers: " + concise
    prompt = (
        "Answer the question using only the evidence passage. Give a concise comma-separated list "
        "and place [1] immediately after every answer. Include all supported answers and do not repeat answers.\n\n"
        f"[1] {context}\n\nQuestion: {question}\nAnswer:"
    )
    return {
        "id": f"multispanqa-{raw['id']}", "question": question, "prompt": prompt,
        "docs": [{"id": 1, "title": "", "text": context}], "edges": edges,
        "short": concise, "long": long, "negative_drop": negative_drop,
        "negative_swap": negative_drop, "negative_redundant": negative_redundant,
    }


def select_by_hash(rows: list[dict], limit: int, seed: int, split: str) -> list[dict]:
    if not limit or limit >= len(rows):
        return rows
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{split}:{row['id']}".encode()).digest())[:limit]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--train-limit", type=int, default=350)
    parser.add_argument("--dev-limit", type=int, default=160)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    mapping = {"train": "train", "dev": "valid", "test": "test"}
    manifest = {
        "dataset": "MultiSpanQA official multi-span split",
        "selection": "smallest SHA256(seed:split:id); official test retained in full",
        "seed": args.seed, "source_sha256": {}, "splits": {},
    }
    for split, source_name in mapping.items():
        source = args.raw_dir / f"{source_name}.json"
        raw_rows = json.loads(source.read_text(encoding="utf-8"))["data"]
        limit = args.train_limit if split == "train" else args.dev_limit if split == "dev" else 0
        chosen = select_by_hash(raw_rows, limit, args.seed, split)
        converted = [convert(row, rng) for row in chosen]
        output = args.out_dir / f"{split}.jsonl"
        write_jsonl(output, converted)
        manifest["source_sha256"][source_name] = sha256(source)
        manifest["splits"][split] = {
            "source_n": len(raw_rows), "n": len(converted),
            "mean_edges": sum(len(row["edges"]) for row in converted) / len(converted),
            "output_sha256": sha256(output),
            "id_sha256": hashlib.sha256("\n".join(row["id"] for row in converted).encode()).hexdigest(),
        }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

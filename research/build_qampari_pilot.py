#!/usr/bin/env python3
"""Build a disjoint QAMPARI pilot for evidence-equivalent concise generation.

The official ALCE retrieval files annotate, for every retrieved document, which
gold answer groups occur in that document.  We use those annotations to create
claim--evidence edges without an LLM judge.  This is a feasibility pilot: the
held-out split is disjoint, but results should not be reported as the official
ALCE benchmark because examples from its evaluation collection are used for
training here.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def clean(text: str) -> str:
    return " ".join(text.replace("\ufffd", " ").split())


def choose_docs_and_edges(item: dict, max_docs: int, max_answers: int):
    """Greedily choose documents that add answer coverage."""
    answers = item["answers"]
    chosen = []
    covered: dict[int, int] = {}
    remaining = list(enumerate(item["docs"]))
    while remaining and len(chosen) < max_docs and len(covered) < max_answers:
        best_pos, best_gain = None, -1
        for pos, (_, doc) in enumerate(remaining):
            found = doc.get("answers_found") or []
            gain = sum(
                1 for idx, present in enumerate(found[: len(answers)])
                if present and idx not in covered
            )
            if gain > best_gain:
                best_pos, best_gain = pos, gain
        if best_pos is None or best_gain <= 0:
            break
        _, doc = remaining.pop(best_pos)
        chosen.append(doc)
        new_doc_id = len(chosen)
        for idx, present in enumerate((doc.get("answers_found") or [])[: len(answers)]):
            if present and idx not in covered and len(covered) < max_answers:
                covered[idx] = new_doc_id
    edges = []
    for answer_idx in sorted(covered):
        aliases = answers[answer_idx]
        if aliases:
            edges.append({
                "answer_idx": answer_idx,
                "answer": clean(aliases[0]),
                "aliases": [clean(x) for x in aliases],
                "doc_id": covered[answer_idx],
            })
    return chosen, edges


def short_target(edges: list[dict]) -> str:
    return ", ".join(f"{e['answer']} [{e['doc_id']}]" for e in edges) + "."


def long_target(edges: list[dict]) -> str:
    parts = []
    for i, edge in enumerate(edges):
        lead = "The retrieved evidence identifies" if i == 0 else "The evidence also identifies"
        parts.append(f"{lead} {edge['answer']} as a relevant answer [{edge['doc_id']}].")
    recap = ", ".join(edge["answer"] for edge in edges)
    return " ".join(parts) + f" In summary, the supported answers are {recap}."


def dropped_target(edges: list[dict], rng: random.Random) -> str:
    if len(edges) <= 1:
        return "I cannot identify all supported answers."
    drop = rng.randrange(len(edges))
    return short_target([e for i, e in enumerate(edges) if i != drop])


def swapped_target(edges: list[dict], n_docs: int, rng: random.Random) -> str:
    bad = [dict(e) for e in edges]
    pos = rng.randrange(len(bad))
    correct = bad[pos]["doc_id"]
    choices = [i for i in range(1, n_docs + 1) if i != correct]
    if not choices:
        return dropped_target(edges, rng)
    bad[pos]["doc_id"] = rng.choice(choices)
    return short_target(bad)


def redundant_target(edges: list[dict]) -> str:
    concise = short_target(edges)
    return (
        "Based on the retrieved documents, the answer can be determined from the evidence. "
        + concise
        + " To repeat the supported answers: "
        + concise
    )


def prompt_for(question: str, docs: list[dict]) -> str:
    rendered = []
    for i, doc in enumerate(docs, 1):
        rendered.append(f"[{i}] {clean(doc.get('title', ''))}: {clean(doc.get('text', ''))}")
    return (
        "Answer the question using only the retrieved web documents. "
        "Give a concise comma-separated list and place the supporting document number "
        "immediately after each answer. Do not cite unsupported documents.\n\n"
        + "\n".join(rendered)
        + f"\n\nQuestion: {clean(question)}\nAnswer:"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-docs", type=int, default=8)
    ap.add_argument("--max-answers", type=int, default=8)
    ap.add_argument("--min-answers", type=int, default=3)
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--test", type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = []
    for item in raw:
        docs, edges = choose_docs_and_edges(item, args.max_docs, args.max_answers)
        if len(edges) < args.min_answers or len(docs) < 2:
            continue
        row = {
            "id": str(item.get("id", item.get("sample_id", len(rows)))),
            "question": clean(item["question"]),
            "prompt": prompt_for(item["question"], docs),
            "docs": [{"id": i + 1, "title": clean(d.get("title", "")), "text": clean(d.get("text", ""))}
                     for i, d in enumerate(docs)],
            "edges": edges,
            "short": short_target(edges),
            "long": long_target(edges),
        }
        row["negative_drop"] = dropped_target(edges, rng)
        row["negative_swap"] = swapped_target(edges, len(docs), rng)
        row["negative_redundant"] = redundant_target(edges)
        rows.append(row)

    rng.shuffle(rows)
    needed = args.train + args.dev + args.test
    if len(rows) < needed:
        raise SystemExit(f"Only {len(rows)} eligible examples; need {needed}")
    rows = rows[:needed]
    splits = {
        "train": rows[: args.train],
        "dev": rows[args.train : args.train + args.dev],
        "test": rows[args.train + args.dev :],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, examples in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    ids = {k: [x["id"] for x in v] for k, v in splits.items()}
    assert not (set(ids["train"]) & set(ids["dev"]))
    assert not (set(ids["train"]) & set(ids["test"]))
    assert not (set(ids["dev"]) & set(ids["test"]))
    stats = {
        "source_examples": len(raw),
        "eligible_examples": len(rows),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "mean_edges": {k: sum(len(x["edges"]) for x in v) / len(v) for k, v in splits.items()},
        "seed": args.seed,
        "warning": "Feasibility split from ALCE evaluation collection; not an official benchmark result.",
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

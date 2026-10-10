#!/usr/bin/env python3
"""Build disjoint ASQA evidence-graph splits for A-EQUAL training."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def clean(text: str) -> str:
    return " ".join((text or "").replace("\ufffd", " ").split())


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", clean(text).lower()).split())


def contains_alias(text: str, aliases: list[str]) -> bool:
    haystack = f" {normalize(text)} "
    return any(f" {normalize(alias)} " in haystack for alias in aliases if len(normalize(alias)) >= 3)


def document_text(doc: dict) -> str:
    return " ".join(
        clean(doc.get(key, "")) for key in ("title", "text", "summary", "extraction")
    )


def answer_groups(item: dict, max_answers: int) -> list[dict]:
    groups = []
    for qa in item.get("qa_pairs", []):
        aliases = [clean(value) for value in qa.get("short_answers", []) if clean(value)]
        if not aliases:
            continue
        groups.append({
            "subquestion": clean(qa.get("question", "")),
            "answer": aliases[0],
            "aliases": aliases,
        })
        if len(groups) >= max_answers:
            break
    return groups


def choose_docs_and_edges(item: dict, max_docs: int, max_answers: int):
    groups = answer_groups(item, max_answers)
    remaining = list(item.get("docs", []))
    chosen = []
    covered: dict[int, int] = {}
    while remaining and len(chosen) < max_docs and len(covered) < len(groups):
        best_pos, best_new, best_score = None, [], float("-inf")
        for pos, doc in enumerate(remaining):
            text = document_text(doc)
            newly_found = [
                idx for idx, group in enumerate(groups)
                if idx not in covered and contains_alias(text, group["aliases"])
            ]
            score = float(doc.get("score", 0.0) or 0.0)
            if len(newly_found) > len(best_new) or (
                len(newly_found) == len(best_new) and score > best_score
            ):
                best_pos, best_new, best_score = pos, newly_found, score
        if best_pos is None or not best_new:
            break
        doc = remaining.pop(best_pos)
        chosen.append(doc)
        doc_id = len(chosen)
        for idx in best_new:
            covered[idx] = doc_id

    edges = []
    for idx in sorted(covered):
        group = groups[idx]
        edges.append({
            "answer_idx": idx,
            "subquestion": group["subquestion"],
            "answer": group["answer"],
            "aliases": group["aliases"],
            "doc_id": covered[idx],
        })
    return chosen, edges


def short_target(edges: list[dict]) -> str:
    return " ".join(
        f'The answer to "{edge["subquestion"]}" is {edge["answer"]} [{edge["doc_id"]}].'
        for edge in edges
    )


def long_target(edges: list[dict]) -> str:
    concise = short_target(edges)
    recap = "; ".join(edge["answer"] for edge in edges)
    return (
        "The retrieved sources support several distinct aspects of the question. "
        + concise
        + f" Taken together, the supported answers are {recap}."
    )


def dropped_target(edges: list[dict], rng: random.Random) -> str:
    if len(edges) <= 1:
        return "The retrieved evidence does not support every aspect of the question."
    drop = rng.randrange(len(edges))
    return short_target([edge for idx, edge in enumerate(edges) if idx != drop])


def swapped_target(edges: list[dict], n_docs: int, rng: random.Random) -> str:
    bad = [dict(edge) for edge in edges]
    pos = rng.randrange(len(bad))
    correct = bad[pos]["doc_id"]
    alternatives = [doc_id for doc_id in range(1, n_docs + 1) if doc_id != correct]
    if not alternatives:
        return dropped_target(edges, rng)
    bad[pos]["doc_id"] = rng.choice(alternatives)
    return short_target(bad)


def redundant_target(edges: list[dict]) -> str:
    concise = short_target(edges)
    return concise + " To restate the same evidence-supported points, " + concise


def prompt_for(question: str, docs: list[dict]) -> str:
    rendered = [
        f"[{idx}] {clean(doc.get('title', ''))}: {clean(doc.get('text', ''))}"
        for idx, doc in enumerate(docs, 1)
    ]
    return (
        "Answer the question using only the retrieved web documents. Cover the distinct "
        "aspects needed for a complete answer, but be concise. Place the supporting "
        "document number immediately after each factual answer. Do not cite unsupported "
        "documents.\n\n"
        + "\n".join(rendered)
        + f"\n\nQuestion: {clean(question)}\nAnswer:"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--max-docs", type=int, default=8)
    ap.add_argument("--max-answers", type=int, default=5)
    ap.add_argument("--min-answers", type=int, default=2)
    ap.add_argument("--train", type=int, default=350)
    ap.add_argument("--dev", type=int, default=80)
    ap.add_argument("--test", type=int, default=159)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    eligible = []
    for item in raw:
        docs, edges = choose_docs_and_edges(item, args.max_docs, args.max_answers)
        if len(edges) < args.min_answers or len(docs) < 2:
            continue
        row = {
            "id": str(item.get("sample_id", len(eligible))),
            "question": clean(item.get("question", "")),
            "prompt": prompt_for(item.get("question", ""), docs),
            "docs": [
                {"id": idx + 1, "title": clean(doc.get("title", "")), "text": clean(doc.get("text", ""))}
                for idx, doc in enumerate(docs)
            ],
            "edges": edges,
            "short": short_target(edges),
            "long": long_target(edges),
        }
        row["negative_drop"] = dropped_target(edges, rng)
        row["negative_swap"] = swapped_target(edges, len(docs), rng)
        row["negative_redundant"] = redundant_target(edges)
        eligible.append(row)

    rng.shuffle(eligible)
    needed = args.train + args.dev + args.test
    if len(eligible) < needed:
        raise SystemExit(f"Only {len(eligible)} eligible examples; need {needed}")
    selected = eligible[:needed]
    splits = {
        "train": selected[: args.train],
        "dev": selected[args.train : args.train + args.dev],
        "test": selected[args.train + args.dev :],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    assert split_ids["train"].isdisjoint(split_ids["dev"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["dev"].isdisjoint(split_ids["test"])
    manifest = {
        "source": args.input,
        "source_examples": len(raw),
        "eligible_examples": len(eligible),
        "selected_examples": needed,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "mean_edges": {
            name: sum(len(row["edges"]) for row in rows) / len(rows)
            for name, rows in splits.items()
        },
        "mean_docs": {
            name: sum(len(row["docs"]) for row in rows) / len(rows)
            for name, rows in splits.items()
        },
        "seed": args.seed,
        "warning": "Disjoint feasibility split derived from the ALCE ASQA evaluation collection; not an official leaderboard result.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

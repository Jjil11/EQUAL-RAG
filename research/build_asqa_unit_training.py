#!/usr/bin/env python3
"""Build a compute-matched evidence-unit training view of ASQA.

Exactly one evidence edge is selected per parent training example, preserving
the number of optimizer steps. Evaluation continues to use the untouched
full-answer ASQA development/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_index(parent_id: str, seed: int, n: int) -> int:
    digest = hashlib.sha256(f"{seed}:{parent_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def render_unit(edge: dict) -> str:
    return (
        f'The answer to "{edge["subquestion"]}" is '
        f'{edge["answer"]} [{edge["doc_id"]}].'
    )


def transform(row: dict, seed: int) -> dict:
    edges = row["edges"]
    pos = stable_index(str(row["id"]), seed, len(edges))
    edge = dict(edges[pos])
    correct = int(edge["doc_id"])
    alternatives = [int(doc["id"]) for doc in row["docs"] if int(doc["id"]) != correct]
    if not alternatives:
        raise ValueError(f'No citation-swap alternative for {row["id"]}')
    swap_pos = stable_index(f'{row["id"]}:swap', seed, len(alternatives))
    swapped = dict(edge)
    swapped["doc_id"] = alternatives[swap_pos]
    chosen = render_unit(edge)
    base_prompt = row["prompt"]
    if base_prompt.rstrip().endswith("Answer:"):
        base_prompt = base_prompt.rstrip()[: -len("Answer:")].rstrip()
    focused_prompt = (
        base_prompt
        + "\n\nRequired evidence unit: answer only the following aspect in one concise "
          "sentence and include its supporting citation.\nAspect: "
        + edge["subquestion"]
        + "\nAnswer:"
    )
    return {
        "id": f'{row["id"]}::unit-{edge["answer_idx"]}',
        "parent_id": str(row["id"]),
        "selected_edge_position": pos,
        "question": row["question"],
        "prompt": focused_prompt,
        "docs": row["docs"],
        "edges": [edge],
        "short": chosen,
        "long": chosen + " This is the evidence-supported answer to the required aspect.",
        "negative_drop": "The retrieved documents do not provide a supported answer to this required aspect.",
        "negative_swap": render_unit(swapped),
        "negative_redundant": chosen + " To restate the same point, " + chosen,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()

    source = read_jsonl(args.input)
    transformed = [transform(row, args.seed) for row in source]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in transformed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_positions = {}
    for row in transformed:
        key = str(row["selected_edge_position"])
        selected_positions[key] = selected_positions.get(key, 0) + 1
    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "parent_examples": len(source),
        "unit_examples": len(transformed),
        "compute_matched_one_unit_per_parent": True,
        "unique_parent_ids": len({row["parent_id"] for row in transformed}),
        "selected_edge_position_counts": selected_positions,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

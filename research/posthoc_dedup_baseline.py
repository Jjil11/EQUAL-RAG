#!/usr/bin/env python3
"""Create and score a deterministic exact-answer-unit de-duplication baseline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_unit(unit: str) -> str:
    return re.sub(r"\s+", " ", unit.strip().lower().rstrip(". ;"))


def deduplicate(text: str) -> str:
    kept, seen = [], set()
    for raw in text.split(","):
        key = normalize_unit(raw)
        if key and key not in seen:
            kept.append(raw.strip())
            seen.add(key)
    return ", ".join(kept)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def contains_alias(text: str, aliases: list[str]) -> bool:
    normalized = f" {normalize_text(text)} "
    return any(f" {normalize_text(alias)} " in normalized for alias in aliases if normalize_text(alias))


def score(text: str, row: dict, tokens: int) -> dict:
    found = grounded = cited = 0
    for edge in row["edges"]:
        if contains_alias(text, edge["aliases"]):
            found += 1
            positions = [text.lower().find(alias.lower()) for alias in edge["aliases"] if alias.lower() in text.lower()]
            positions = [position for position in positions if position >= 0]
            if positions:
                window = text[min(positions): min(positions) + 160]
                refs = {int(value) for value in re.findall(r"\[(\d+)\]", window)}
                if refs:
                    cited += 1
                if edge["doc_id"] in refs:
                    grounded += 1
    refs = [int(value) for value in re.findall(r"\[(\d+)\]", text)]
    return {
        "id": row["id"], "output": text, "tokens": tokens,
        "gold_edges": len(row["edges"]), "answers_found": found,
        "answers_cited": cited, "grounded_edges": grounded,
        "citations": len(refs), "valid_citations": sum(1 for value in refs if 1 <= value <= len(row["docs"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = read_jsonl(args.input)
    data = {str(row["id"]): row for row in read_jsonl(args.data)}
    if set(str(row["id"]) for row in outputs) != set(data):
        raise ValueError("Evaluation IDs and dataset IDs differ")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    results = []
    changed = 0
    for output in outputs:
        text = deduplicate(output["output"])
        changed += text != output["output"]
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        results.append(score(text, data[str(output["id"])], tokens))
    edges = sum(row["gold_edges"] for row in results)
    mean_tokens = sum(row["tokens"] for row in results) / len(results)
    quality = sum(row["grounded_edges"] for row in results) / max(1, edges)
    metrics = {
        "n": len(results), "changed_outputs": changed,
        "answer_recall": sum(row["answers_found"] for row in results) / max(1, edges),
        "citation_coverage": sum(row["answers_cited"] for row in results) / max(1, edges),
        "grounded_edge_recall": quality,
        "citation_validity": sum(row["valid_citations"] for row in results) / max(1, sum(row["citations"] for row in results)),
        "mean_output_tokens": mean_tokens,
        "grounded_utility_per_token": quality / max(1, mean_tokens),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "eval_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

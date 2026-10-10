"""Build answer-blind short/long distillation sets matched within each question."""

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--audit", action="append", required=True)
    parser.add_argument("--short-out", required=True)
    parser.add_argument("--long-out", required=True)
    parser.add_argument("--min-gap", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    source = {}
    for idx, original in enumerate(read_jsonl(args.source_data)):
        row = dict(original)
        row.setdefault("idx", idx)
        answer = str(row.get("answer", ""))
        row.setdefault("gold", answer.split("####")[-1].strip())
        row.setdefault("natural_trace", answer.split("####")[0].strip())
        source[str(row["idx"])] = row

    audits = []
    for path in args.audit:
        audits.extend(read_jsonl(path))

    pairs = []
    for audit in audits:
        # Keep formatting quality matched across arms. Numeric fallback extraction can mark
        # an unboxed candidate correct, but such a target violates the training/eval contract.
        correct = [
            candidate
            for candidate in audit["candidates"]
            if candidate["correct"] and "\\boxed" in candidate.get("text", "")
        ]
        if len(correct) < 2:
            continue
        correct.sort(key=lambda candidate: (candidate["completion_tokens"], candidate["text"]))
        short, long = correct[0], correct[-1]
        gap = long["completion_tokens"] - short["completion_tokens"]
        if gap < args.min_gap:
            continue
        base = source.get(str(audit["idx"]))
        if base is None:
            continue
        pairs.append((base, short, long, gap))

    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    short_rows, long_rows = [], []
    for base, short, long, gap in pairs:
        common = {
            **base,
            "pair_gap_tokens": gap,
            "pair_rule": "same question and answer-blind sampling request; shortest versus longest correct candidate",
            "answer_blind": True,
        }
        short_rows.append({
            **common,
            "cap_target": short["text"],
            "cap_completion_tokens": short["completion_tokens"],
            "pair_arm": "short",
        })
        long_rows.append({
            **common,
            "cap_target": long["text"],
            "cap_completion_tokens": long["completion_tokens"],
            "pair_arm": "long",
        })

    short_path, long_path = Path(args.short_out), Path(args.long_out)
    write_jsonl(short_path, short_rows)
    write_jsonl(long_path, long_rows)
    summary = {
        "pairs": len(pairs),
        "min_gap": args.min_gap,
        "short_mean_tokens": mean(short_rows, "cap_completion_tokens"),
        "long_mean_tokens": mean(long_rows, "cap_completion_tokens"),
        "mean_within_question_gap": mean(short_rows, "pair_gap_tokens"),
        "same_question_ids": [row["idx"] for row in short_rows] == [row["idx"] for row in long_rows],
        "generation": "answer-blind; gold used only for post-hoc correctness filtering",
        "target_format": "both arms require \\boxed output",
        "seed": args.seed,
    }
    for path in (short_path.parent / "SUMMARY.json", long_path.parent / "SUMMARY.json"):
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

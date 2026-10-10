"""Create a strict paired dataset with matched formatting and optional token-order checks."""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", required=True)
    parser.add_argument("--long", required=True)
    parser.add_argument("--short-out", required=True)
    parser.add_argument("--long-out", required=True)
    parser.add_argument("--require-boxed", action="store_true")
    args = parser.parse_args()

    short_rows = {str(row["idx"]): row for row in read_jsonl(args.short)}
    long_rows = {str(row["idx"]): row for row in read_jsonl(args.long)}
    kept_short, kept_long = [], []
    rejected = {"missing_arm": 0, "metadata_mismatch": 0, "format": 0, "order": 0}
    for idx in sorted(set(short_rows) | set(long_rows), key=lambda value: int(value)):
        if idx not in short_rows or idx not in long_rows:
            rejected["missing_arm"] += 1
            continue
        short, long = short_rows[idx], long_rows[idx]
        if str(short.get("question")) != str(long.get("question")) or str(short.get("gold")) != str(long.get("gold")):
            rejected["metadata_mismatch"] += 1
            continue
        if args.require_boxed and ("\\boxed" not in short.get("cap_target", "") or "\\boxed" not in long.get("cap_target", "")):
            rejected["format"] += 1
            continue
        if int(short["cap_completion_tokens"]) >= int(long["cap_completion_tokens"]):
            rejected["order"] += 1
            continue
        kept_short.append(short)
        kept_long.append(long)

    write_jsonl(args.short_out, kept_short)
    write_jsonl(args.long_out, kept_long)
    summary = {
        "input_short": len(short_rows),
        "input_long": len(long_rows),
        "kept_pairs": len(kept_short),
        "rejected": rejected,
        "require_boxed": args.require_boxed,
        "same_ids": [row["idx"] for row in kept_short] == [row["idx"] for row in kept_long],
        "mean_stored_gap": sum(
            int(long["cap_completion_tokens"]) - int(short["cap_completion_tokens"])
            for short, long in zip(kept_short, kept_long)
        ) / len(kept_short),
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    for path in (Path(args.short_out).parent, Path(args.long_out).parent):
        (path / "SANITIZE_SUMMARY.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

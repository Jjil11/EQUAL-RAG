"""Audit paired short/long distillation data before cross-model experiments."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_question(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def fingerprint(text):
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", required=True)
    parser.add_argument("--long", required=True)
    parser.add_argument("--test")
    parser.add_argument("--anchors")
    parser.add_argument("--tokenizer", help="Optional HF tokenizer for cross-tokenizer length audit")
    parser.add_argument("--out")
    args = parser.parse_args()

    short_rows, long_rows = read_jsonl(args.short), read_jsonl(args.long)
    short_by_id = {str(row["idx"]): row for row in short_rows}
    long_by_id = {str(row["idx"]): row for row in long_rows}
    shared = sorted(set(short_by_id) & set(long_by_id))
    errors = []

    if len(short_by_id) != len(short_rows):
        errors.append("duplicate idx values in short data")
    if len(long_by_id) != len(long_rows):
        errors.append("duplicate idx values in long data")
    if set(short_by_id) != set(long_by_id):
        errors.append("short/long idx sets differ")

    field_mismatches = Counter()
    stored_gaps = []
    for idx in shared:
        s, l = short_by_id[idx], long_by_id[idx]
        for field in ("question", "gold"):
            if str(s.get(field, "")) != str(l.get(field, "")):
                field_mismatches[field] += 1
        if s.get("pair_arm") != "short" or l.get("pair_arm") != "long":
            field_mismatches["pair_arm"] += 1
        if not s.get("answer_blind") or not l.get("answer_blind"):
            field_mismatches["answer_blind"] += 1
        sg = int(s.get("cap_completion_tokens", -1))
        lg = int(l.get("cap_completion_tokens", -1))
        stored_gaps.append(lg - sg)
        if sg >= lg:
            field_mismatches["stored_length_order"] += 1
        if s.get("cap_target") == l.get("cap_target"):
            field_mismatches["identical_target"] += 1
        if "\\boxed" not in str(s.get("cap_target", "")):
            field_mismatches["short_missing_boxed"] += 1
        if "\\boxed" not in str(l.get("cap_target", "")):
            field_mismatches["long_missing_boxed"] += 1

    short_questions = [fingerprint(row.get("question", "")) for row in short_rows]
    duplicate_questions = len(short_questions) - len(set(short_questions))
    test_overlap = None
    if args.test:
        test_questions = {fingerprint(row.get("question") or row.get("problem") or "") for row in read_jsonl(args.test)}
        test_overlap = len(set(short_questions) & test_questions)

    anchor_summary = None
    if args.anchors:
        anchors = read_jsonl(args.anchors)
        anchor_ids = {str(row.get("idx")) for row in anchors}
        anchor_questions = {fingerprint(row.get("question", "")) for row in anchors}
        anchor_summary = {
            "rows": len(anchors),
            "unique_ids": len(anchor_ids),
            "overlap_pair_ids": len(anchor_ids & set(shared)),
            "overlap_pair_questions": len(anchor_questions & set(short_questions)),
        }

    tokenizer_summary = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        gaps = []
        for idx in shared:
            short_len = len(tokenizer.encode(short_by_id[idx]["cap_target"], add_special_tokens=False))
            long_len = len(tokenizer.encode(long_by_id[idx]["cap_target"], add_special_tokens=False))
            gaps.append(long_len - short_len)
        tokenizer_summary = {
            "name": args.tokenizer,
            "shorter": sum(gap > 0 for gap in gaps),
            "equal": sum(gap == 0 for gap in gaps),
            "reversed": sum(gap < 0 for gap in gaps),
            "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
            "min_gap": min(gaps) if gaps else None,
            "max_gap": max(gaps) if gaps else None,
        }

    report = {
        "short_rows": len(short_rows),
        "long_rows": len(long_rows),
        "shared_ids": len(shared),
        "field_mismatches": dict(field_mismatches),
        "duplicate_normalized_questions": duplicate_questions,
        "stored_mean_gap": sum(stored_gaps) / len(stored_gaps) if stored_gaps else 0.0,
        "stored_min_gap": min(stored_gaps) if stored_gaps else None,
        "test_question_overlap": test_overlap,
        "anchors": anchor_summary,
        "cross_tokenizer": tokenizer_summary,
        "errors": errors,
        "usable": not errors and not field_mismatches and duplicate_questions == 0 and test_overlap in (None, 0),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

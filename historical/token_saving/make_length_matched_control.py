"""Create disjoint, size- and length-matched certified/control datasets."""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--certified", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--matched-certified-out-dir")
    args = parser.parse_args()

    pool = read_jsonl(Path(args.pool))
    certified = read_jsonl(Path(args.certified))
    certified_ids = {str(row["idx"]) for row in certified}
    uncertified = [row for row in pool if str(row["idx"]) not in certified_ids]

    # In one dimension, an optimal absolute-distance matching preserves order.
    # Dynamic programming chooses a subset from the larger group when counts differ.
    def optimal_subset_match(left, right):
        left = sorted(left, key=lambda row: (row["cap_completion_tokens"], str(row["idx"])))
        right = sorted(right, key=lambda row: (row["cap_completion_tokens"], str(row["idx"])))
        if len(left) > len(right):
            matched_right, matched_left = optimal_subset_match(right, left)
            return matched_left, matched_right
        m, n = len(left), len(right)
        inf = float("inf")
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        take = [[False] * (n + 1) for _ in range(m + 1)]
        for j in range(n + 1):
            dp[0][j] = 0.0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                skip = dp[i][j - 1]
                pair = dp[i - 1][j - 1] + abs(
                    left[i - 1]["cap_completion_tokens"] - right[j - 1]["cap_completion_tokens"]
                )
                if pair <= skip:
                    dp[i][j], take[i][j] = pair, True
                else:
                    dp[i][j] = skip
        chosen = []
        i, j = m, n
        while i:
            if take[i][j]:
                chosen.append(right[j - 1])
                i -= 1
            j -= 1
        chosen.reverse()
        return left, chosen

    selected, matched_certified = optimal_subset_match(uncertified, certified)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.matched_certified_out_dir:
        certified_out = Path(args.matched_certified_out_dir)
        certified_out.mkdir(parents=True, exist_ok=True)
        with (certified_out / "train.jsonl").open("w", encoding="utf-8") as handle:
            for row in matched_certified:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    mean = lambda rows: sum(row["cap_completion_tokens"] for row in rows) / len(rows) if rows else 0.0
    summary = {
        "records": len(selected),
        "control": "uncertified answer-blind candidates greedily matched on completion length",
        "certified_available": len(certified),
        "uncertified_available": len(uncertified),
        "certified_mean_tokens": mean(matched_certified),
        "control_mean_tokens": mean(selected),
        "certified_ids_excluded": True,
        "matching": "minimum total absolute token difference with optimal 1D subset selection",
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.matched_certified_out_dir:
        (certified_out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

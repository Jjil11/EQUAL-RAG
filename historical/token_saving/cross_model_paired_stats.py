#!/usr/bin/env python3
"""Paired short-vs-long statistics for one benchmark evaluation."""

import argparse
import json
import math
from pathlib import Path

from scipy.stats import binomtest, wilcoxon


def load(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return {str(row.get("idx", row.get("id"))): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--long", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    short = load(args.short)
    long = load(args.long)
    ids = sorted(short.keys() & long.keys())
    if not ids:
        raise SystemExit("no paired result IDs")

    wins = sum(bool(short[i]["correct"]) and not bool(long[i]["correct"]) for i in ids)
    losses = sum(bool(long[i]["correct"]) and not bool(short[i]["correct"]) for i in ids)
    short_acc = sum(bool(short[i]["correct"]) for i in ids) / len(ids)
    long_acc = sum(bool(long[i]["correct"]) for i in ids) / len(ids)
    accuracy_deltas = [float(bool(short[i]["correct"])) - float(bool(long[i]["correct"])) for i in ids]
    accuracy_delta = short_acc - long_acc
    accuracy_variance = sum((x - accuracy_delta) ** 2 for x in accuracy_deltas) / max(1, len(ids) - 1)
    accuracy_se = math.sqrt(accuracy_variance / len(ids))
    short_tokens = [float(short[i]["completion_tokens"]) for i in ids]
    long_tokens = [float(long[i]["completion_tokens"]) for i in ids]
    deltas = [a - b for a, b in zip(short_tokens, long_tokens)]
    mean_delta = sum(deltas) / len(deltas)
    variance = sum((x - mean_delta) ** 2 for x in deltas) / max(1, len(deltas) - 1)
    se = math.sqrt(variance / len(deltas))
    nonzero = [x for x in deltas if x != 0]
    token_p = float(wilcoxon(nonzero).pvalue) if nonzero else 1.0
    result = {
        "n": len(ids),
        "short_accuracy": short_acc,
        "long_accuracy": long_acc,
        "accuracy_delta_short_minus_long": accuracy_delta,
        "accuracy_delta_95ci_normal": [accuracy_delta - 1.96 * accuracy_se, accuracy_delta + 1.96 * accuracy_se],
        "short_mean_completion_tokens": sum(short_tokens) / len(ids),
        "long_mean_completion_tokens": sum(long_tokens) / len(ids),
        "token_delta_short_minus_long": mean_delta,
        "relative_token_delta_vs_long": mean_delta / (sum(long_tokens) / len(ids)),
        "token_delta_95ci_normal": [mean_delta - 1.96 * se, mean_delta + 1.96 * se],
        "shorter_equal_longer": [
            sum(x < 0 for x in deltas),
            sum(x == 0 for x in deltas),
            sum(x > 0 for x in deltas),
        ],
        "discordant_short_wins": wins,
        "discordant_long_wins": losses,
        "mcnemar_exact_p": float(binomtest(min(wins, losses), wins + losses, 0.5).pvalue)
        if wins + losses
        else 1.0,
        "wilcoxon_token_p": token_p,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

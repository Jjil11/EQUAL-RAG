#!/usr/bin/env python3
"""Paired bootstrap for interleaved adapter latency records."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def metric(rows: list[dict], field: str, samples: int, seed: int) -> dict:
    differences = [row["method"][field] - row["baseline"][field] for row in rows]
    method = statistics.fmean(row["method"][field] for row in rows)
    baseline = statistics.fmean(row["baseline"][field] for row in rows)
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(differences[rng.randrange(len(rows))] for _ in rows) for _ in range(samples))
    return {
        "n": len(rows), "method": method, "baseline": baseline, "delta": method - baseline,
        "relative_delta": method / baseline - 1,
        "ci95": [boot[int(0.025 * samples)], boot[min(samples - 1, int(0.975 * samples))]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [row for row in data["records"] if not row["warmup"]]
    result = {
        "seconds": metric(rows, "seconds", args.samples, args.seed),
        "generated_tokens": metric(rows, "generated_tokens", args.samples, args.seed + 1),
        "order_balance": {
            "baseline_first": sum(row["order"][0] == "baseline" for row in rows),
            "method_first": sum(row["order"][0] == "method" for row in rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

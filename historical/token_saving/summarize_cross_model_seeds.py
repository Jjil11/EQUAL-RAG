#!/usr/bin/env python3
"""Summarize paired short-vs-long effects across independent training seeds."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


MODELS = {
    "mistral7b": "Mistral-7B-Instruct-v0.3",
    "phi4mini": "Phi-4-mini-instruct",
}


def mean_sd(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def pm(mean: float, sample_sd: float | None, scale: float = 1.0, digits: int = 2) -> str:
    mean_text = f"{scale * mean:+.{digits}f}"
    if sample_sd is None:
        return mean_text + " (one seed; SD unavailable)"
    return mean_text + f" ± {scale * sample_sd:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    output: dict[str, object] = {
        "unit_of_replication": "independent training seed",
        "warning": (
            "Test examples are shared across seeds. Seed-level mean and sample SD are primary; "
            "examples are not pooled across seeds as independent observations."
        ),
        "models": {},
    }
    md = [
        "# Cross-model multi-seed summary",
        "",
        "The unit of replication is the independent training seed. All runs share the same GSM8K test set, so results are summarized at seed level rather than pseudo-replicating test examples across seeds.",
        "",
    ]
    tidy_rows = []

    for prefix, display_name in MODELS.items():
        filename_re = re.compile(
            rf"^{re.escape(prefix)}_strict_gap16_cap500_seed(\d+)_paired_stats\.json$"
        )
        paths = sorted(
            path
            for path in args.input_dir.glob(f"{prefix}_strict_gap16_cap500_seed*_paired_stats.json")
            if filename_re.fullmatch(path.name)
        )
        if not paths:
            continue
        rows = []
        for path in paths:
            stats = json.loads(path.read_text(encoding="utf-8"))
            match = filename_re.fullmatch(path.name)
            assert match is not None
            seed = int(match.group(1))
            rows.append({"seed": seed, "path": str(path), **stats})
            tidy_rows.append(
                {
                    "model": display_name,
                    "seed": seed,
                    "n": stats["n"],
                    "short_accuracy": stats["short_accuracy"],
                    "long_accuracy": stats["long_accuracy"],
                    "accuracy_delta_short_minus_long": stats[
                        "accuracy_delta_short_minus_long"
                    ],
                    "short_mean_completion_tokens": stats[
                        "short_mean_completion_tokens"
                    ],
                    "long_mean_completion_tokens": stats[
                        "long_mean_completion_tokens"
                    ],
                    "token_delta_short_minus_long": stats[
                        "token_delta_short_minus_long"
                    ],
                    "relative_token_delta_vs_long": stats[
                        "relative_token_delta_vs_long"
                    ],
                    "mcnemar_exact_p": stats["mcnemar_exact_p"],
                    "wilcoxon_token_p": stats["wilcoxon_token_p"],
                }
            )

        metrics = {
            "short_accuracy": mean_sd([r["short_accuracy"] for r in rows]),
            "long_accuracy": mean_sd([r["long_accuracy"] for r in rows]),
            "accuracy_delta_short_minus_long": mean_sd(
                [r["accuracy_delta_short_minus_long"] for r in rows]
            ),
            "short_mean_completion_tokens": mean_sd(
                [r["short_mean_completion_tokens"] for r in rows]
            ),
            "long_mean_completion_tokens": mean_sd(
                [r["long_mean_completion_tokens"] for r in rows]
            ),
            "token_delta_short_minus_long": mean_sd(
                [r["token_delta_short_minus_long"] for r in rows]
            ),
            "relative_token_delta_vs_long": mean_sd(
                [r["relative_token_delta_vs_long"] for r in rows]
            ),
        }
        direction = {
            "token_saving_seeds": sum(r["token_delta_short_minus_long"] < 0 for r in rows),
            "short_accuracy_higher_seeds": sum(
                r["accuracy_delta_short_minus_long"] > 0 for r in rows
            ),
            "long_accuracy_higher_seeds": sum(
                r["accuracy_delta_short_minus_long"] < 0 for r in rows
            ),
            "accuracy_ties": sum(
                math.isclose(r["accuracy_delta_short_minus_long"], 0.0) for r in rows
            ),
        }
        output["models"][prefix] = {
            "display_name": display_name,
            "num_seeds": len(rows),
            "seeds": rows,
            "seed_level_summary": metrics,
            "direction_consistency": direction,
        }

        md.extend([
            f"## {display_name}",
            "",
            "| Seed | Short acc. | Long acc. | Acc. delta | Short tokens | Long tokens | Token delta | Relative saving | McNemar p | Token p |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            md.append(
                f"| {row['seed']} | {100 * row['short_accuracy']:.2f}% | "
                f"{100 * row['long_accuracy']:.2f}% | "
                f"{100 * row['accuracy_delta_short_minus_long']:+.2f} pp | "
                f"{row['short_mean_completion_tokens']:.2f} | "
                f"{row['long_mean_completion_tokens']:.2f} | "
                f"{row['token_delta_short_minus_long']:+.2f} | "
                f"{-100 * row['relative_token_delta_vs_long']:.2f}% | "
                f"{row['mcnemar_exact_p']:.3g} | {row['wilcoxon_token_p']:.3g} |"
            )
        a = metrics["accuracy_delta_short_minus_long"]
        t = metrics["token_delta_short_minus_long"]
        rel = metrics["relative_token_delta_vs_long"]
        md.extend([
            "",
            f"Across {len(rows)} seed(s): accuracy delta = {pm(a['mean'], a['sample_sd'], 100)} pp; token delta = {pm(t['mean'], t['sample_sd'])}; relative saving = {pm(-rel['mean'], rel['sample_sd'], 100)} pp (mean ± sample SD when available).",
            f"Short supervision reduced mean completion length in {direction['token_saving_seeds']}/{len(rows)} seeds; long supervision had higher accuracy in {direction['long_accuracy_higher_seeds']}/{len(rows)} seeds.",
            "",
        ])

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_out.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tidy_rows[0]))
            writer.writeheader()
            writer.writerows(tidy_rows)


if __name__ == "__main__":
    main()

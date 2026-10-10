#!/usr/bin/env python3
from __future__ import annotations
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models

import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(str(_release_root))
REPORT = ROOT / "reports"

RUNS = {
    "mistral": [
        {
            "seed": 20260930,
            "dev": ROOT / "runs/qampari_trajectory_onpolicy_repair_seed20260930/interp05/dev_paired_vs_aequal.json",
            "test": ROOT / "runs/qampari_trajectory_onpolicy_repair_seed20260930/interp05/regression_test_paired_vs_aequal.json",
        },
        {
            "seed": 20261001,
            "root": ROOT / "runs/qampari_onpolicy_multiseed/seed20261001",
        },
        {
            "seed": 20261002,
            "root": ROOT / "runs/qampari_onpolicy_multiseed/seed20261002",
        },
    ],
    "qwen3_14b": [
        {
            "seed": 20260929,
            "root": ROOT / "runs/qwen3_14b_onpolicy_multiseed/seed20260929",
        },
        {
            "seed": 20260930,
            "root": ROOT / "runs/qwen3_14b_onpolicy_repair_seed20260930",
        },
        {
            "seed": 20261001,
            "root": ROOT / "runs/qwen3_14b_onpolicy_multiseed/seed20261001",
        },
    ],
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_record(path: Path):
    data = load(path)
    paired = data["paired_method_minus_baseline"]
    quality = paired["grounded_edge_recall"]
    tokens = paired["output_tokens"]
    utility = data["grounded_utility_per_token"]
    trunc = data.get("truncation_rate_256", {})
    return {
        "quality_method": quality["mean_a"],
        "quality_baseline": quality["mean_b"],
        "quality_delta": quality["delta_a_minus_b"],
        "quality_ci95": quality["ci95"],
        "quality_p": quality["p_two_sided"],
        "tokens_method": tokens["mean_a"],
        "tokens_baseline": tokens["mean_b"],
        "tokens_delta": tokens["delta_a_minus_b"],
        "tokens_ci95": tokens["ci95"],
        "tokens_p": tokens["p_two_sided"],
        "utility_method": utility["mean_a"],
        "utility_baseline": utility["mean_b"],
        "utility_delta": utility["delta_a_minus_b"],
        "utility_relative_delta": utility["relative_delta"],
        "utility_ci95": utility["ci95"],
        "utility_p": utility["p_two_sided"],
        "truncation_method": trunc.get("method"),
        "truncation_baseline": trunc.get("baseline"),
        "quality_noninferior_margin_0p01": quality["delta_a_minus_b"] >= -0.01,
        "tokens_improved": tokens["delta_a_minus_b"] < 0,
        "utility_improved": utility["relative_delta"] > 0,
    }


def stages(run_root: Path):
    path = run_root / "stage_history.tsv"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stamp, stage = line.split("\t", 1)
        rows.append((datetime.fromisoformat(stamp), stage))
    durations = {}
    for (start, stage), (end, _) in zip(rows, rows[1:]):
        durations[stage.lower()] = (end - start).total_seconds()
    if rows:
        durations["total"] = (rows[-1][0] - rows[0][0]).total_seconds()
    return durations


def summarize(records):
    fields = ["quality_delta", "tokens_delta", "utility_relative_delta"]
    out = {"n_seeds": len(records)}
    for field in fields:
        values = [record[field] for record in records]
        out[field] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    out["quality_noninferior_count"] = sum(r["quality_noninferior_margin_0p01"] for r in records)
    out["tokens_improved_count"] = sum(r["tokens_improved"] for r in records)
    out["utility_improved_count"] = sum(r["utility_improved"] for r in records)
    return out


def main():
    report = {"protocol": "frozen hyperparameters, interpolation weight 0.5, cap 256", "models": {}}
    for model, specs in RUNS.items():
        model_rows = []
        for spec in specs:
            row = {"seed": spec["seed"], "splits": {}}
            for split in ("dev", "test"):
                if split in spec:
                    path = spec[split]
                else:
                    path = spec["root"] / "interp05" / f"{split}_paired_vs_aequal.json"
                row["splits"][split] = paired_record(path)
            if "root" in spec:
                mining = spec["root"] / "mining/mining_summary.json"
                row["mining"] = load(mining) if mining.exists() else None
                row["stage_seconds"] = stages(spec["root"])
                manifest = spec["root"] / "repair/train_manifest.json"
                row["train_seconds"] = load(manifest).get("elapsed_train_seconds") if manifest.exists() else None
            model_rows.append(row)
        report["models"][model] = {
            "seeds": model_rows,
            "aggregate_dev": summarize([row["splits"]["dev"] for row in model_rows]),
            "aggregate_test": summarize([row["splits"]["test"] for row in model_rows]),
        }

    REPORT.mkdir(parents=True, exist_ok=True)
    json_path = REPORT / "onpolicy_batch_summary_20261005.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# On-policy repair batch summary (2026-10-05)", ""]
    for model, payload in report["models"].items():
        lines += [f"## {model}", "", "| Seed | Split | Δ grounded quality | Δ tokens | Relative Δ utility | Q noninferior |", "|---:|---|---:|---:|---:|:---:|"]
        for row in payload["seeds"]:
            for split in ("dev", "test"):
                x = row["splits"][split]
                lines.append(f"| {row['seed']} | {split} | {x['quality_delta']:+.6f} | {x['tokens_delta']:+.3f} | {100*x['utility_relative_delta']:+.2f}% | {'yes' if x['quality_noninferior_margin_0p01'] else 'no'} |")
        for split in ("dev", "test"):
            agg = payload[f"aggregate_{split}"]
            lines += ["", f"{split} seed mean ± sample SD: quality {agg['quality_delta']['mean']:+.6f} ± {agg['quality_delta']['sample_std']:.6f}; tokens {agg['tokens_delta']['mean']:+.3f} ± {agg['tokens_delta']['sample_std']:.3f}; relative utility {100*agg['utility_relative_delta']['mean']:+.2f}% ± {100*agg['utility_relative_delta']['sample_std']:.2f}%. Noninferior {agg['quality_noninferior_count']}/{agg['n_seeds']}; token improved {agg['tokens_improved_count']}/{agg['n_seeds']}; utility improved {agg['utility_improved_count']}/{agg['n_seeds']}.", ""]
    md_path = REPORT / "onpolicy_batch_summary_20261005.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "models": report["models"]}, indent=2))


if __name__ == "__main__":
    main()

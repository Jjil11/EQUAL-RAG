#!/usr/bin/env python3
"""Aggregate the frozen dynamic-budget baseline over all model/seed runs."""

from __future__ import annotations
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models

import argparse
import json
import statistics
import subprocess
from pathlib import Path


ROOT = Path(str(_release_root))

RUNS = [
    ("mistral", 20260930,
     "runs/qampari_sequal_seed20260930/dev_aequal_cap256", "runs/qampari_cap256_seed20260930/aequal_kl",
     "runs/qampari_trajectory_onpolicy_repair_seed20260930/interp05/dev", "runs/qampari_trajectory_onpolicy_repair_seed20260930/interp05/regression_test"),
    ("mistral", 20261001,
     "runs/qampari_onpolicy_multiseed/seed20261001/baseline_dev", "runs/qampari_onpolicy_multiseed/seed20261001/baseline_test",
     "runs/qampari_onpolicy_multiseed/seed20261001/interp05/dev", "runs/qampari_onpolicy_multiseed/seed20261001/interp05/test"),
    ("mistral", 20261002,
     "runs/qampari_onpolicy_multiseed/seed20261002/baseline_dev", "runs/qampari_onpolicy_multiseed/seed20261002/baseline_test",
     "runs/qampari_onpolicy_multiseed/seed20261002/interp05/dev", "runs/qampari_onpolicy_multiseed/seed20261002/interp05/test"),
    ("qwen3", 20260930,
     "runs/qwen3_14b_onpolicy_repair_seed20260930/baseline_dev", "runs/qwen3_14b_onpolicy_repair_seed20260930/baseline_test",
     "runs/qwen3_14b_onpolicy_repair_seed20260930/interp05/dev", "runs/qwen3_14b_onpolicy_repair_seed20260930/interp05/test"),
    ("qwen3", 20261001,
     "runs/qwen3_14b_onpolicy_multiseed/seed20261001/baseline_dev", "runs/qwen3_14b_onpolicy_multiseed/seed20261001/baseline_test",
     "runs/qwen3_14b_onpolicy_multiseed/seed20261001/interp05/dev", "runs/qwen3_14b_onpolicy_multiseed/seed20261001/interp05/test"),
    ("qwen3", 20260929,
     "runs/qwen3_14b_onpolicy_multiseed/seed20260929/baseline_dev", "runs/qwen3_14b_onpolicy_multiseed/seed20260929/baseline_test",
     "runs/qwen3_14b_onpolicy_multiseed/seed20260929/interp05/dev", "runs/qwen3_14b_onpolicy_multiseed/seed20260929/interp05/test"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def deltas(method: dict, baseline: dict) -> dict:
    q = method["grounded_edge_recall"] - baseline["grounded_edge_recall"]
    t = method["mean_output_tokens"] - baseline["mean_output_tokens"]
    u = method["grounded_utility_per_token"] / baseline["grounded_utility_per_token"] - 1
    return {"quality": q, "tokens": t, "utility_relative": u}


def aggregate(items: list[dict], comparison: str) -> dict:
    ds = [item[comparison] for item in items]
    result = {}
    for key in ("quality", "tokens", "utility_relative"):
        values = [d[key] for d in ds]
        result[key] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
        }
    result.update({
        "quality_noninferior_count": sum(d["quality"] >= -0.01 for d in ds),
        "tokens_lower_count": sum(d["tokens"] < 0 for d in ds),
        "utility_higher_count": sum(d["utility_relative"] > 0 for d in ds),
        "n": len(ds),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    analyzer = ROOT / "code/analyze_paired_outputs.py"
    rows = []
    for family, seed, base_dev, base_test, equal_dev, equal_test in RUNS:
        run = ROOT / f"runs/dynamic_budget_baseline_20261006/{family}_seed{seed}"
        item = {"family": family, "seed": seed, "splits": {}}
        for split, base_rel, equal_rel in (("dev", base_dev, equal_dev), ("test", base_test, equal_test)):
            dynamic_dir = run / split
            pair_equal = run / f"{split}_paired_vs_equal.json"
            subprocess.run([
                str(ROOT / "../miniforge3/envs/puma/bin/python"), str(analyzer),
                "--method", str(dynamic_dir / "eval_results.jsonl"),
                "--baseline", str(ROOT / equal_rel / "eval_results.jsonl"),
                "--output", str(pair_equal), "--samples", "20000", "--seed", str(seed),
                "--truncation-cap", "256",
            ], check=True)
            dynamic = load(dynamic_dir / "metrics.json")
            base = load(ROOT / base_rel / "metrics.json")
            equal = load(ROOT / equal_rel / "metrics.json")
            repetition = load(run / f"{split}_repetition.json")
            item["splits"][split] = {
                "dynamic": dynamic, "plain_aequal": base, "equal_rag": equal,
                "dynamic_minus_plain": deltas(dynamic, base),
                "dynamic_minus_equal": deltas(dynamic, equal),
                "paired_vs_plain": load(run / f"{split}_paired_vs_plain_aequal.json"),
                "paired_vs_equal": load(pair_equal),
                "repetition": repetition,
            }
        rows.append(item)
    report = {
        "protocol": "q95 training-target-tokens-per-retrieved-document; input-only dynamic prompt budget; cap 256",
        "runs": rows,
        "aggregate_test": {
            "all_vs_plain": aggregate([r["splits"]["test"] for r in rows], "dynamic_minus_plain"),
            "all_vs_equal": aggregate([r["splits"]["test"] for r in rows], "dynamic_minus_equal"),
            "mistral_vs_plain": aggregate([r["splits"]["test"] for r in rows if r["family"] == "mistral"], "dynamic_minus_plain"),
            "qwen3_vs_plain": aggregate([r["splits"]["test"] for r in rows if r["family"] == "qwen3"], "dynamic_minus_plain"),
        },
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate_test"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controlled offline vLLM replay for WCLP short/long LoRA adapters.

The engine, prompts, sampling parameters, and GPU remain fixed.  Both adapters
are loaded into the same vLLM process, warmed up, and then evaluated in an
alternating randomized order.  This avoids comparing the incomparable timing
numbers emitted by earlier runs that executed on different/concurrent GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from eval_cap_official import BOX, chat, gold, read_jsonl


def nvml_readers():
    try:
        from vllm.third_party import pynvml

        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        # NVML indexes physical devices and is not remapped by CUDA visibility.
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible))
        memory = lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**30
        try:
            # A100-class devices expose a cumulative millijoule counter.
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            energy = lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
        except Exception:
            energy = None
        return energy, memory
    except Exception:
        return None, None


def summarize(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--short-adapter", required=True)
    parser.add_argument("--long-adapter", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=1319)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--warmup-n", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        parser.error("--rounds must be an even integer >= 2 for counterbalancing")

    source = read_jsonl(Path(args.data))[: args.n]
    llm = LLM(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        enforce_eager=True,
        seed=0,
        enable_lora=True,
        max_lora_rank=16,
        # Replaying identical prompts across arms with prefix caching enabled
        # would advantage whichever arm runs second. Disable it for comparison.
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    instruction = (
        f"Solve with compact derivation under about {args.max_tokens} tokens. "
        "Use equations or short factual steps only. Preserve constraints and substitutions; "
        f"avoid prose. End with {BOX} and no text after it."
    )
    prompts = [chat(tokenizer, instruction + "\n\nProblem:\n" + row["question"]) for row in source]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    requests = {
        "short": LoRARequest("wclp_short", 1, args.short_adapter),
        "long": LoRARequest("wclp_long", 2, args.long_adapter),
    }

    # Warm both LoRA paths before timing; warmup output is discarded.
    for arm in ("short", "long"):
        llm.generate(prompts[: args.warmup_n], params, lora_request=requests[arm], use_tqdm=False)

    rows = []
    read_energy, read_memory = nvml_readers()
    for round_id in range(1, args.rounds + 1):
        # Strictly counterbalance order: SL, LS, SL, LS.  An even number of
        # rounds ensures each arm runs first equally often.
        order = ["short", "long"] if round_id % 2 else ["long", "short"]
        for order_index, arm in enumerate(order):
            energy_start = read_energy() if read_energy else None
            started = time.perf_counter()
            outputs = llm.generate(prompts, params, lora_request=requests[arm], use_tqdm=False)
            elapsed = time.perf_counter() - started
            energy_end = read_energy() if read_energy else None
            completion_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            rows.append({
                "round": round_id,
                "order_index": order_index,
                "arm": arm,
                "requests": len(prompts),
                "elapsed_seconds": elapsed,
                "request_throughput": len(prompts) / elapsed,
                "output_token_throughput": completion_tokens / elapsed,
                "completion_tokens": completion_tokens,
                "mean_completion_tokens": completion_tokens / len(prompts),
                "resident_gpu_memory_gib": read_memory() if read_memory else None,
                "energy_joules": (energy_end - energy_start) if energy_start is not None else None,
            })

    by_arm = {}
    for arm in ("short", "long"):
        group = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {
            "elapsed_seconds": summarize([row["elapsed_seconds"] for row in group]),
            "request_throughput": summarize([row["request_throughput"] for row in group]),
            "output_token_throughput": summarize([row["output_token_throughput"] for row in group]),
            "mean_completion_tokens": summarize([row["mean_completion_tokens"] for row in group]),
        }
        memories = [row["resident_gpu_memory_gib"] for row in group if row["resident_gpu_memory_gib"] is not None]
        by_arm[arm]["resident_gpu_memory_gib"] = summarize(memories) if memories else None
        energies = [row["energy_joules"] for row in group if row["energy_joules"] is not None]
        by_arm[arm]["energy_joules"] = summarize(energies) if energies else None

    short, long = by_arm["short"], by_arm["long"]
    paired = []
    for round_id in range(1, args.rounds + 1):
        s = next(row for row in rows if row["round"] == round_id and row["arm"] == "short")
        l = next(row for row in rows if row["round"] == round_id and row["arm"] == "long")
        item = {
            "round": round_id,
            "request_throughput_gain_fraction": s["request_throughput"] / l["request_throughput"] - 1,
            "wall_time_reduction_fraction": 1 - s["elapsed_seconds"] / l["elapsed_seconds"],
            "completion_token_reduction_fraction": 1 - s["mean_completion_tokens"] / l["mean_completion_tokens"],
        }
        if s["energy_joules"] is not None and l["energy_joules"] is not None:
            item["energy_reduction_fraction"] = 1 - s["energy_joules"] / l["energy_joules"]
        paired.append(item)
    comparison = {
        key: summarize([row[key] for row in paired])
        for key in (
            "request_throughput_gain_fraction",
            "wall_time_reduction_fraction",
            "completion_token_reduction_fraction",
        )
    }
    energy_pairs = [row["energy_reduction_fraction"] for row in paired if "energy_reduction_fraction" in row]
    if energy_pairs:
        comparison["energy_reduction_fraction"] = summarize(energy_pairs)

    payload = {
        "design": "single fixed vLLM engine; prefix caching disabled; both LoRAs warmed; strictly counterbalanced within-round arm order",
        "model": args.model,
        "n": len(prompts),
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "raw_rounds": rows,
        "paired_round_comparisons": paired,
        "summary": by_arm,
        "comparison": comparison,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def pct(value: float) -> str:
        return f"{100 * value:+.2f}%"

    report = [
        "# Controlled vLLM replay",
        "",
        f"Model: `{args.model}`; requests per arm: {len(prompts)}; rounds: {args.rounds}.",
        "",
        "| Arm | Mean completion | Wall time (s) | Requests/s | Output tokens/s | Resident GPU GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("short", "long"):
        item = by_arm[arm]
        report.append(
            f"| {arm} | {item['mean_completion_tokens']['mean']:.2f} | "
            f"{item['elapsed_seconds']['mean']:.2f} +/- {item['elapsed_seconds']['sample_sd']:.2f} | "
            f"{item['request_throughput']['mean']:.2f} | {item['output_token_throughput']['mean']:.2f} | "
            f"{item['resident_gpu_memory_gib']['mean']:.2f} |"
        )
    report.extend([
        "",
        f"Short-minus-long request-throughput change: {pct(comparison['request_throughput_gain_fraction']['mean'])} +/- {100*comparison['request_throughput_gain_fraction']['sample_sd']:.2f} pp.",
        f"Short-relative wall-time reduction: {pct(comparison['wall_time_reduction_fraction']['mean'])} +/- {100*comparison['wall_time_reduction_fraction']['sample_sd']:.2f} pp.",
        f"Short-relative completion-token reduction: {pct(comparison['completion_token_reduction_fraction']['mean'])} +/- {100*comparison['completion_token_reduction_fraction']['sample_sd']:.2f} pp.",
        "",
        "This is a saturated offline batch replay. It does not estimate TTFT or online tail latency.",
    ])
    if "energy_reduction_fraction" in comparison:
        report.insert(-1, f"Short-relative GPU energy reduction: {pct(comparison['energy_reduction_fraction']['mean'])} +/- {100*comparison['energy_reduction_fraction']['sample_sd']:.2f} pp.")
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()

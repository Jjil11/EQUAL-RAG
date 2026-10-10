"""Evaluate CAP on an unseen benchmark split with fixed, matched controls."""

import argparse
import json
import os
import re
from pathlib import Path

# A legacy server workaround masked physical GPU 2 from NVML. It must be
# opt-in: CUDA_VISIBLE_DEVICES=2 is a valid single-GPU evaluation assignment.
if os.environ.get("SOLAR_MASK_NVML_GPU2") == "1":
    try:
        import vllm.third_party.pynvml as pynvml
        original_handle, original_name = pynvml.nvmlDeviceGetHandleByIndex, pynvml.nvmlDeviceGetName
        pynvml.nvmlDeviceGetHandleByIndex = lambda index: original_handle(index) if index != 2 else None
        pynvml.nvmlDeviceGetName = lambda handle: original_name(handle) if handle is not None else "Unavailable GPU"
    except Exception:
        pass

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


BASE = Path(os.environ.get("SOLAR_BASE", "/private-storage/solar-token-experiments"))
BOX = r"\boxed{answer}"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold(row):
    return str(row["answer"]).split("####")[-1].strip()


def norm(value):
    return re.sub(r"\s+", "", str(value).replace(",", "").replace("\\%", "%").strip("$ "))


def extract(text):
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    values = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    return values[-1] if values else ""


def correct(prediction, answer):
    prediction, answer = norm(prediction), norm(answer)
    if prediction == answer:
        return True
    try:
        return abs(float(prediction.rstrip("%")) - float(answer.rstrip("%"))) < 1e-6
    except ValueError:
        return False


def chat(tokenizer, content):
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def cd_instruction():
    return f"Solve with compact derivation under about 192 tokens. Use equations or short factual steps only. Preserve constraints and substitutions; avoid prose. End with {BOX} and no text after it."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--data", default=str(BASE / "dataset/benchmarks/gsm8k_test.jsonl"))
    parser.add_argument("--adapter", default=str(BASE / "runs/qwen3_14b_cap_counterfactual_pilot/adapter"))
    parser.add_argument("--out-dir", default=str(BASE / "runs/qwen3_14b_cap_counterfactual_pilot/eval_gsm8k_test"))
    parser.add_argument("--method-name", default="cap_cf", help="Label used for the adapted variants in reports")
    parser.add_argument("--n", type=int, default=0, help="0 evaluates every example")
    parser.add_argument("--only-192", action="store_true", help="Skip the 128-token budget variant for seed replication")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Use the native Transformers/vLLM architecture")
    args = parser.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    items = [{"idx": idx, "question": row["question"], "gold": gold(row)}
             for idx, row in enumerate(read_jsonl(Path(args.data)))]
    if args.n:
        items = items[:args.n]
    variants = [("base_cd_192", 192, None),
                (f"{args.method_name}_cd_192", 192, LoRARequest(args.method_name, 1, args.adapter)),
                (f"{args.method_name}_cd_128", 128, LoRARequest(args.method_name, 1, args.adapter))]
    if args.only_192:
        variants = variants[:2]
    llm = LLM(model=args.model, trust_remote_code=not args.no_trust_remote_code, tensor_parallel_size=1,
              gpu_memory_utilization=0.82, max_model_len=8192, enforce_eager=True, seed=0,
              enable_lora=True, max_lora_rank=16)
    tokenizer = llm.get_tokenizer()
    prompts = [chat(tokenizer, cd_instruction() + "\n\nProblem:\n" + item["question"]) for item in items]
    results = []
    for name, limit, request in variants:
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=limit, seed=0), lora_request=request)
        for item, prompt, output in zip(items, prompts, outputs):
            text = output.outputs[0].text
            completion = len(output.outputs[0].token_ids)
            prediction = extract(text)
            results.append({**item, "method": name, "limit": limit,
                            "prompt_tokens": len(tokenizer.encode(prompt)), "completion_tokens": completion,
                            "total_tokens": len(tokenizer.encode(prompt)) + completion,
                            "correct": correct(prediction, item["gold"]), "truncated": completion >= limit,
                            "prediction": prediction, "text": text})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = ["# CAP Counterfactual Official Evaluation", "", "| method | n | accuracy | truncation | avg prompt | avg completion | avg total |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, _, _ in variants:
        group = [row for row in results if row["method"] == name]
        avg = lambda key: sum(row[key] for row in group) / len(group)
        lines.append(f"| {name} | {len(group)} | {avg('correct'):.4f} | {avg('truncated'):.4f} | {avg('prompt_tokens'):.1f} | {avg('completion_tokens'):.1f} | {avg('total_tokens'):.1f} |")
    lines.extend(["", "All rows come from the benchmark split specified by --data; no training indices are present."])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "REPORT.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

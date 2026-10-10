"""Run the adapted half of eval_cap_official without regenerating the base."""

import argparse
import json
import os
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from eval_cap_official import BOX, chat, correct, extract, gold, read_jsonl
from eval_solar_cd_lora_cross_dataset import math_equal, svamp_equal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--dataset-name", choices=("GSM8K", "MATH500", "SVAMP"), default="GSM8K")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    source = read_jsonl(Path(args.data))
    if args.dataset_name == "GSM8K":
        items = [{"idx": idx, "question": row["question"], "gold": gold(row)} for idx, row in enumerate(source)]
        judge = correct
    else:
        items = [
            {
                "idx": idx,
                "question": row.get("problem") or row.get("question") or "",
                "gold": str(row["answer"]).strip(),
            }
            for idx, row in enumerate(source)
        ]
        judge = math_equal if args.dataset_name == "MATH500" else svamp_equal
    llm = LLM(
        model=args.model,
        trust_remote_code=not args.no_trust_remote_code,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.82,
        max_model_len=8192,
        enforce_eager=True,
        seed=0,
        enable_lora=True,
        max_lora_rank=16,
    )
    tokenizer = llm.get_tokenizer()
    instruction = (
        f"Solve with compact derivation under about {args.max_tokens} tokens. "
        f"Use equations or short factual steps only. Preserve constraints and substitutions; "
        f"avoid prose. End with {BOX} and no text after it."
    )
    prompts = [chat(tokenizer, instruction + "\n\nProblem:\n" + item["question"]) for item in items]
    request = LoRARequest(args.method_name, 1, args.adapter)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0),
        lora_request=request,
    )
    rows = []
    for item, prompt, output in zip(items, prompts, outputs):
        text = output.outputs[0].text
        completion = len(output.outputs[0].token_ids)
        prediction = extract(text)
        prompt_tokens = len(tokenizer.encode(prompt))
        rows.append(
            {
                **item,
                "method": f"{args.method_name}_cd_{args.max_tokens}",
                "limit": args.max_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion,
                "total_tokens": prompt_tokens + completion,
                "correct": judge(prediction, item["gold"]),
                "truncated": completion >= args.max_tokens,
                "prediction": prediction,
                "text": text,
            }
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    avg = lambda key: sum(row[key] for row in rows) / len(rows)
    report = (
        "# Adapter-only Official Evaluation\n\n"
        "| method | n | accuracy | truncation | avg prompt | avg completion | avg total |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"| {args.method_name}_cd_{args.max_tokens} | {len(rows)} | {avg('correct'):.4f} | "
        f"{avg('truncated'):.4f} | {avg('prompt_tokens'):.1f} | "
        f"{avg('completion_tokens'):.1f} | {avg('total_tokens'):.1f} |\n"
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check whether loading an adapter second changes greedy generations."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model, tokenizer, rows, max_length, max_new_tokens, disable_thinking):
    out = {}
    for index, row in enumerate(rows):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=not disable_thinking,
        )
        inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, inputs["input_ids"].shape[1]:].cpu().tolist()
        payload = ",".join(map(str, generated)).encode()
        out[str(row["id"])] = {"tokens": len(generated), "sha256": hashlib.sha256(payload).hexdigest()}
        print(json.dumps({"done": index + 1, "total": len(rows)}), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--baseline-adapter", required=True)
    p.add_argument("--method-adapter", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--disable-thinking", action="store_true")
    args = p.parse_args()
    all_rows = [json.loads(x) for x in args.data.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {str(x["id"]): x for x in all_rows}
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["records"]
    measured = [x for x in reference if not x.get("warmup")]
    long_ids = [str(x["id"]) for x in sorted(measured, key=lambda x: x["generated_tokens"], reverse=True)[: args.n // 2]]
    short_ids = [str(x["id"]) for x in measured if str(x["id"]) not in set(long_ids)][: args.n - len(long_ids)]
    rows = [by_id[x] for x in long_ids + short_ids]
    tokenizer = AutoTokenizer.from_pretrained(args.method_adapter, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    primary_model = PeftModel.from_pretrained(base, args.method_adapter, adapter_name="method").cuda().eval()
    primary = generate(primary_model, tokenizer, rows, args.max_length, args.max_new_tokens, args.disable_thinking)
    del primary_model, base
    gc.collect(); torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, local_files_only=True)
    secondary_model = PeftModel.from_pretrained(base, args.baseline_adapter, adapter_name="baseline")
    secondary_model.load_adapter(args.method_adapter, adapter_name="method")
    secondary_model.set_adapter("method")
    secondary_model = secondary_model.cuda().eval()
    secondary = generate(secondary_model, tokenizer, rows, args.max_length, args.max_new_tokens, args.disable_thinking)
    comparisons = []
    for row in rows:
        key = str(row["id"])
        comparisons.append({"id": key, "primary": primary[key], "secondary": secondary[key],
                            "exact": primary[key]["sha256"] == secondary[key]["sha256"]})
    report = {"n": len(comparisons), "exact_count": sum(x["exact"] for x in comparisons),
              "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n": report["n"], "exact_count": report["exact_count"]}, indent=2))


if __name__ == "__main__":
    main()

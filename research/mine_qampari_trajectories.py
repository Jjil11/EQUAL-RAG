#!/usr/bin/env python3
"""Mine real repetition-onset states from frozen QAMPARI model trajectories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_unit(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def split_units(output: str, mode: str) -> list[str]:
    if mode == "comma":
        return output.split(",")
    if mode == "citation_sentence":
        units = []
        start = 0
        for match in re.finditer(r"\[\d+\]\.(?:\s+|$)", output):
            units.append(output[start:match.end()].strip())
            start = match.end()
        if start < len(output) and output[start:].strip():
            units.append(output[start:].strip())
        return units
    if mode == "line":
        return [unit for unit in output.splitlines() if unit.strip()]
    raise ValueError(mode)


def first_repetition(output: str, min_chars: int = 2, mode: str = "comma"):
    raw_units = split_units(output, mode)
    seen: dict[str, int] = {}
    for index, raw in enumerate(raw_units):
        normalized = normalize_unit(raw)
        if len(normalized) < min_chars:
            continue
        if normalized in seen and index > 0:
            separator = {"comma": ",", "citation_sentence": " ", "line": "\n"}[mode]
            prefix = separator.join(raw_units[:index]).strip().rstrip(" ,")
            continuation = separator + raw
            if prefix and normalize_unit(continuation):
                return {
                    "trajectory_prefix": prefix,
                    "trajectory_continuation": continuation,
                    "trajectory_repeat_unit": normalized,
                    "trajectory_first_index": index,
                    "trajectory_source_index": seen[normalized],
                }
        else:
            seen[normalized] = index
    return None


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--unit-mode", choices=["comma", "citation_sentence", "line"], default="comma")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    selected = [
        (index, row) for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter).cuda().eval()
    model.config.use_cache = True

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    mined = 0
    cap_hits = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for local_index, (global_index, row) in enumerate(selected, start=1):
            template_kwargs = {
                "enable_thinking": not args.disable_thinking,
            }
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            encoded = tokenizer(prefix, return_tensors="pt").to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = generated[0, encoded["input_ids"].shape[1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            token_count = int(new_ids.numel())
            cap_hit = token_count >= args.max_new_tokens
            trajectory = first_repetition(output, mode=args.unit_mode)
            result = {
                "global_index": global_index,
                "id": row["id"],
                "output": output,
                "tokens": token_count,
                "cap_hit": cap_hit,
                "has_trajectory": trajectory is not None,
            }
            if trajectory:
                result.update(trajectory)
                mined += 1
            cap_hits += int(cap_hit)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if local_index % 10 == 0 or local_index == len(selected):
                print(json.dumps({
                    "shard": args.shard_index,
                    "done": local_index,
                    "total": len(selected),
                    "mined": mined,
                    "cap_hits": cap_hits,
                }), flush=True)
    temporary.replace(output_path)


if __name__ == "__main__":
    main()

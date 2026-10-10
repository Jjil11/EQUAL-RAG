"""Generate compact answer-blind reasoning candidates and keep shortest correct traces.

The model sees only the problem. Gold answers are used strictly after generation
for outcome filtering, which avoids answer-conditioned backward rationalization.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from vllm import LLM, SamplingParams


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chat(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def norm(value: str) -> str:
    return re.sub(r"\s+", "", str(value).replace(",", "").strip("$ "))


def extract(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    values = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    return values[-1] if values else ""


def matches(text: str, gold: str) -> bool:
    prediction, answer = norm(extract(text)), norm(gold)
    if prediction == answer:
        return True
    try:
        return abs(float(prediction) - float(answer)) < 1e-6
    except ValueError:
        return False


def prompt(question: str, target_tokens: int) -> str:
    return (
        "Solve the problem independently using a compact but complete natural mathematical derivation. "
        "Keep only necessary equations, substitutions, and short factual steps; do not restate the problem "
        "or add repeated verification. Do not assume or work backward from any proposed answer. "
        f"Aim for at most {target_tokens} output tokens. End with exactly \\boxed{{answer}} and no text after it.\n\n"
        f"Problem:\n{question}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--target-tokens", type=int, default=72)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    rows = read_jsonl(Path(args.data))
    normalized = []
    for position, original in enumerate(rows):
        row = dict(original)
        row.setdefault("idx", position)
        if "gold" not in row:
            answer = str(row.get("answer", ""))
            row["gold"] = answer.split("####")[-1].strip()
        if "natural_trace" not in row:
            answer = str(row.get("answer", ""))
            row["natural_trace"] = answer.split("####")[0].strip()
        normalized.append(row)
    rows = normalized
    for position, row in enumerate(rows):
        row.setdefault("idx", position)
        if "gold" not in row and "answer" in row:
            row["gold"] = str(row["answer"]).split("####")[-1].strip()
        if "natural_trace" not in row and "answer" in row:
            row["natural_trace"] = str(row["answer"]).split("####")[0].strip()
    random.Random(args.seed).shuffle(rows)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    rows = rows[args.shard_index :: args.num_shards]
    if args.n:
        rows = rows[: args.n]

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.82,
        max_model_len=8192,
        enforce_eager=True,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [chat(tokenizer, prompt(row["question"], args.target_tokens)) for row in rows]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.7,
            top_p=0.95,
            n=args.candidates,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )

    kept, audit = [], []
    for row, request_output in zip(rows, outputs):
        candidates = []
        for output in request_output.outputs:
            text = output.text.strip()
            tokens = len(output.token_ids)
            candidates.append({
                "text": text,
                "completion_tokens": tokens,
                "correct": matches(text, row["gold"]),
                "within_target": tokens <= args.target_tokens,
            })
        eligible = [x for x in candidates if x["correct"] and x["within_target"]]
        eligible.sort(key=lambda x: x["completion_tokens"])
        audit.append({"idx": row["idx"], "candidates": candidates, "accepted": bool(eligible)})
        if not eligible:
            continue
        best = eligible[0]
        source_tokens = len(tokenizer.encode(str(row.get("natural_trace", ""))))
        kept.append({
            **row,
            "cap_target": best["text"],
            "source_completion_tokens": source_tokens,
            "cap_completion_tokens": best["completion_tokens"],
            "cap_reduction": 1 - best["completion_tokens"] / max(source_tokens, 1),
            "cap_rule": "answer-blind generation; shortest exact-correct candidate within target budget",
            "answer_blind": True,
            "candidate_count": args.candidates,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in [("train.jsonl", kept), ("audit.jsonl", audit)]:
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "requested": len(rows),
        "accepted": len(kept),
        "acceptance": len(kept) / len(rows) if rows else 0.0,
        "candidates_per_problem": args.candidates,
        "target_tokens": args.target_tokens,
        "mean_cap_tokens": sum(x["cap_completion_tokens"] for x in kept) / len(kept) if kept else 0.0,
        "generation": "answer-blind: model receives problem only; gold used post hoc",
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

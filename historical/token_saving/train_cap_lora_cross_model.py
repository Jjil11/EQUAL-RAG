"""Capability-anchored LoRA SFT for Counterfactual Answer-Preserving Compression."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup


BASE = Path("/private-storage/solar-token-experiments")
BOX = r"\boxed{answer}"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def instruction(question: str, mode: str) -> str:
    if mode == "bridge":
        return (
            "Solve in a compact verification-first format. State the requested variable once, then keep only "
            "the equations, constraints, substitutions, and final answer needed to derive it. "
            f"Avoid extra explanation. End with {BOX} and no text after it.\n\nProblem:\n{question}"
        )
    return (
        "Solve with a compact derivation. Keep only equations, constraints, substitutions, and the final answer. "
        f"Avoid explanatory prose. End with {BOX} and no text after it.\n\nProblem:\n{question}"
    )


def chat(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError):
            return f"User: {content}\nAssistant:"
    except (AttributeError, ValueError):
        return f"User: {content}\nAssistant:"


def discover_lora_targets(model):
    """Select projection modules across Qwen, Llama/Mistral, Phi, and Gemma families."""
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "qkv_proj", "query_key_value", "dense",
        "gate_proj", "up_proj", "down_proj", "gate_up_proj",
        "fc1", "fc2",
    }
    observed = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = sorted(candidates & observed)
    if not targets:
        raise RuntimeError(f"Could not discover LoRA projection modules; observed examples: {sorted(observed)[:80]}")
    return targets


def collate(batch, pad_id):
    longest = max(item["input_ids"].shape[0] for item in batch)
    ids, labels, masks = [], [], []
    for item in batch:
        length = item["input_ids"].shape[0]
        padding = longest - length
        ids.append(torch.cat([item["input_ids"], torch.full((padding,), pad_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((padding,), -100, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(length, dtype=torch.long), torch.zeros(padding, dtype=torch.long)]))
    return {"input_ids": torch.stack(ids), "labels": torch.stack(labels), "attention_mask": torch.stack(masks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--cap-data", default=str(BASE / "runs/cap_natural_train/train.jsonl"))
    parser.add_argument("--target-field", default="cap_target", help="Verified response field to distill from each compression record")
    parser.add_argument("--instruction-mode", choices=["equation", "bridge"], default="equation")
    parser.add_argument("--pairs", default=str(BASE / "runs/qwen3_14b_solar_cd_ab_v3_pairs_train/pairs.jsonl"))
    parser.add_argument("--out-dir", default=str(BASE / "runs/qwen3_14b_cap_natural_pilot"))
    parser.add_argument("--max-cap-samples", type=int, default=435)
    parser.add_argument("--max-anchor-samples", type=int, default=870)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--allow-overlap-anchors", action="store_true")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Use the native Transformers implementation")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    rng = random.Random(args.seed)
    cap_rows = read_jsonl(Path(args.cap_data))
    if any(args.target_field not in row for row in cap_rows):
        raise KeyError(f"Missing target field {args.target_field!r} in compression data.")
    rng.shuffle(cap_rows)
    cap_rows = cap_rows[: args.max_cap_samples]
    cap_ids = {str(row["idx"]) for row in cap_rows}
    pair_rows = []
    for row in read_jsonl(Path(args.pairs)):
        teacher = row["base_192"]
        disjoint = args.allow_overlap_anchors or str(row["idx"]) not in cap_ids
        if disjoint and teacher["correct"] and teacher["completion_tokens"] < 190 and "\\boxed" in teacher["text"]:
            pair_rows.append({"idx": row["idx"], "question": row["question"], "target": teacher["text"]})
    rng.shuffle(pair_rows)
    anchors = pair_rows[: args.max_anchor_samples]
    examples = [("cap", row["question"], row[args.target_field], row["idx"]) for row in cap_rows]
    examples += [("anchor", row["question"], row["target"], row["idx"]) for row in anchors]
    rng.shuffle(examples)

    trust_remote_code = not args.no_trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = []
    for source, question, target, idx in examples:
        prefix = chat(tokenizer, instruction(question, args.instruction_mode))
        encoded = tokenizer(prefix + target, add_special_tokens=False, truncation=True, max_length=args.max_length, return_tensors="pt")["input_ids"][0]
        prefix_len = min(len(tokenizer(prefix, add_special_tokens=False)["input_ids"]), len(encoded))
        labels = encoded.clone()
        labels[:prefix_len] = -100
        if (labels != -100).sum() >= 2:
            prepared.append({"input_ids": encoded, "labels": labels, "source": source, "idx": idx})
    if not prepared:
        raise RuntimeError("No usable CAP training records.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "records": len(prepared), "cap_records": sum(x["source"] == "cap" for x in prepared), "anchor_records": sum(x["source"] == "anchor" for x in prepared), "target_field": args.target_field, "instruction_mode": args.instruction_mode, "learning_rate": args.learning_rate, "seed": args.seed, "allow_overlap_anchors": args.allow_overlap_anchors, "cap_ids": sorted(cap_ids), "anchor_ids": sorted(str(row["idx"]) for row in anchors), "design": "verified compact traces with disjoint correct-trace capability anchors"}
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[data]", json.dumps(manifest), flush=True)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    target_modules = discover_lora_targets(model)
    manifest["lora_target_modules"] = target_modules
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[lora]", json.dumps(target_modules), flush=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    loader = DataLoader(prepared, batch_size=1, shuffle=True, collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    accumulation = 8
    updates = (len(loader) + accumulation - 1) // accumulation
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, updates // 20), num_training_steps=updates)
    history, micro = [], 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch_id, batch in enumerate(loader, start=1):
        batch = {name: value.cuda() for name, value in batch.items()}
        loss = model(**batch).loss / accumulation
        loss.backward()
        micro += 1
        if micro % accumulation == 0 or batch_id == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            record = {"update": len(history) + 1, "loss": float(loss.item() * accumulation), "lr": float(scheduler.get_last_lr()[0])}
            history.append(record)
            if record["update"] % 10 == 0 or batch_id == len(loader):
                print("[train]", record, flush=True)
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"updates": len(history), "final_loss": history[-1]["loss"], "adapter": str(out_dir / "adapter")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

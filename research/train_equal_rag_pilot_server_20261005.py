#!/usr/bin/env python3
"""LoRA pilot for Long/Short SFT, length preference, and EQUAL-RAG.

EQUAL-RAG uses counterfactual graph violations as rejected responses.  The
pairwise objective is reference-free (SimPO-style) and is combined with CE on
the concise evidence-complete response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Rows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def one(items):
    return items[0]


def anchor_cache_path(cache_dir: str, row_id: str) -> Path:
    digest = hashlib.sha256(str(row_id).encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.pt"


def encode_example(tokenizer, prompt: str, response: str, max_length: int):
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    template_kwargs = {"enable_thinking": not getattr(tokenizer, "_equal_rag_disable_thinking", False)}
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **template_kwargs
    )
    prefix = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    encoded = tokenizer(full, truncation=True, max_length=max_length, return_tensors="pt")
    prefix_ids = tokenizer(prefix, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
    ids = encoded["input_ids"][0]
    mask = encoded["attention_mask"][0]
    labels = ids.clone()
    labels[: min(prefix_ids.shape[1], labels.shape[0])] = -100
    return ids, mask, labels


def response_logp(model, tokenizer, prompt, response, max_length, return_response_logits=False):
    ids, mask, labels = encode_example(tokenizer, prompt, response, max_length)
    device = next(model.parameters()).device
    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
    out = model(input_ids=ids.unsqueeze(0), attention_mask=mask.unsqueeze(0))
    logits = out.logits[:, :-1].float()
    target = labels.unsqueeze(0)[:, 1:]
    valid = target.ne(-100)
    safe = target.masked_fill(~valid, 0)
    token_lp = F.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    mean_logp = (token_lp * valid).sum() / valid.sum().clamp_min(1)
    mean_nll = -mean_logp
    # The final supervised token is the chat-template end-of-turn/EOS token.
    # Boosting this term explicitly teaches the model to stop after the complete
    # evidence-equivalent list rather than entering a repetition loop.
    last_pos = valid[0].nonzero(as_tuple=False)[-1, 0]
    stop_nll = -token_lp[0, last_pos]
    if return_response_logits:
        return mean_logp, mean_nll, stop_nll, logits[0, valid[0]]
    return mean_logp, mean_nll, stop_nll


def evidence_unit_variants(row: dict) -> tuple[str, str, str, str, str]:
    """Sample one edge and construct local completeness/citation/redundancy contrasts."""
    edge = dict(random.choice(row["edges"]))
    correct_doc = int(edge["doc_id"])
    alternatives = [int(doc["id"]) for doc in row["docs"] if int(doc["id"]) != correct_doc]
    if not alternatives:
        raise ValueError(f'No citation-swap alternative for {row["id"]}')
    swapped = dict(edge)
    swapped["doc_id"] = random.choice(alternatives)

    is_decomposed = bool(edge.get("subquestion"))

    def render(item: dict) -> str:
        if is_decomposed:
            return (
                f'The answer to "{item["subquestion"]}" is '
                f'{item["answer"]} [{item["doc_id"]}].'
            )
        return f'{item["answer"]} [{item["doc_id"]}].'

    base_prompt = row["prompt"]
    if base_prompt.rstrip().endswith("Answer:"):
        base_prompt = base_prompt.rstrip()[: -len("Answer:")].rstrip()
    if is_decomposed:
        unit_request = (
            "Answer only the following aspect in one concise sentence and include "
            f'its supporting citation.\nAspect: {edge["subquestion"]}'
        )
    else:
        # List-QA edges have no per-answer subquestion.  The supporting document
        # identifies one unambiguous evidence unit without revealing its answer.
        unit_request = (
            "Return only the single answer item to the original question that is "
            f"supported by document [{correct_doc}], with that citation."
        )
    unit_prompt = base_prompt + "\n\nRequired evidence unit: " + unit_request + "\nAnswer:"
    chosen = render(edge)
    dropped = "The retrieved documents do not provide a supported answer to this required aspect."
    redundant = chosen + " To restate the same point, " + chosen
    return unit_prompt, chosen, dropped, render(swapped), redundant


def _boundary_log_probs(model, tokenizer, prompt: str, response_prefix: str,
                        continuation: str, max_length: int):
    """Return log P(end-of-turn) and log P(first continuation token) at a boundary."""
    ids, mask, labels = encode_example(tokenizer, prompt, response_prefix, max_length)
    valid_positions = labels.ne(-100).nonzero(as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        raise ValueError("Response prefix has no supervised tokens")
    end_pos = int(valid_positions[-1])
    end_token = int(ids[end_pos])
    if end_pos <= 0:
        raise ValueError("Cannot score an empty response boundary")
    context_ids = ids[:end_pos]
    context_mask = mask[:end_pos]
    continuation_ids = tokenizer(
        continuation, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]
    if continuation_ids.numel() == 0:
        raise ValueError("Continuation tokenization is empty")
    device = next(model.parameters()).device
    out = model(
        input_ids=context_ids.unsqueeze(0).to(device),
        attention_mask=context_mask.unsqueeze(0).to(device),
    )
    log_probs = F.log_softmax(out.logits[0, -1].float(), dim=-1)
    non_end = log_probs.clone()
    non_end[end_token] = -torch.inf
    return (
        log_probs[end_token],
        log_probs[int(continuation_ids[0])],
        non_end.max(),
    )


def saturation_boundary_loss(model, tokenizer, row: dict, max_length: int,
                             margin: float):
    """Prefer EOS after complete evidence and a missing unit before completion."""
    edges = list(row["edges"])
    if len(edges) < 2:
        raise ValueError(f'Saturation boundary needs at least two edges: {row["id"]}')
    missing_index = random.randrange(len(edges))
    missing = edges[missing_index]

    def render(edge: dict) -> str:
        return f'{edge["answer"]} [{int(edge["doc_id"])}]'

    # The ordinary stop loss already supervises the clean gold boundary.  The
    # observed failure happens off that path: after one answer unit has already
    # been repeated, generation enters a loop.  Train recovery at exactly that
    # state and require EOS to beat every possible continuation token.
    repeated_prefix = row["short"].rstrip().rstrip(".") + ", " + render(missing) + "."
    duplicate = " " + render(missing) + "."
    full_end_lp, repeat_lp, strongest_continue_lp = _boundary_log_probs(
        model, tokenizer, row["prompt"], repeated_prefix, duplicate, max_length
    )
    complete_loss = F.relu(margin - (full_end_lp - strongest_continue_lp))

    # At an incomplete list boundary, adding the missing unit must beat ending.
    remaining = [render(edge) for i, edge in enumerate(edges) if i != missing_index]
    incomplete_prefix = ", ".join(remaining)
    missing_continuation = ", " + render(missing) + "."
    premature_end_lp, missing_lp, _ = _boundary_log_probs(
        model, tokenizer, row["prompt"], incomplete_prefix,
        missing_continuation, max_length
    )
    incomplete_loss = F.relu(margin - (missing_lp - premature_end_lp))
    return 0.5 * (complete_loss + incomplete_loss), complete_loss, incomplete_loss


def trajectory_boundary_loss(model, tokenizer, row: dict, max_length: int,
                             margin: float):
    """Prefer EOS at a repetition onset actually visited by the frozen teacher."""
    end_lp, continuation_lp, strongest_continue_lp = _boundary_log_probs(
        model,
        tokenizer,
        row["prompt"],
        row["trajectory_prefix"],
        row["trajectory_continuation"],
        max_length,
    )
    # The strongest-token contrast is stricter than supervising only the mined
    # token and prevents the model from replacing one loop token with another.
    loss = F.relu(margin - (end_lp - strongest_continue_lp))
    return loss, end_lp, continuation_lp, strongest_continue_lp


def save_adapter(model, tokenizer, out_dir: Path, manifest: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def train(args, model, tokenizer, rows, anchor_model=None):
    loader = DataLoader(Rows(rows), batch_size=1, shuffle=True, collate_fn=one)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    model.config.use_cache = False
    history = []
    step = 0
    dual_value = args.dual_init
    dual_ema = args.constraint_target
    pending_violations = []
    last_grad_norm = None
    optim.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for row in loader:
            if args.method == "long_sft":
                _, loss, _ = response_logp(model, tokenizer, row["prompt"], row["long"], args.max_length)
            elif args.method in {
                "short_sft", "equal_rag", "length_pref", "short_stop",
                "equal_cov", "equal_cov_stop", "equal_full_stop", "redundancy_stop",
                "cequal_dual", "aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl",
                "trajectory_equal", "trajectory_repair",
            }:
                if args.method in {"aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"}:
                    chosen_lp, chosen_ce, stop_nll, chosen_logits = response_logp(
                        model, tokenizer, row["prompt"], row["short"], args.max_length,
                        return_response_logits=True,
                    )
                else:
                    chosen_lp, chosen_ce, stop_nll = response_logp(
                        model, tokenizer, row["prompt"], row["short"], args.max_length
                    )
                loss = chosen_ce
                if args.method == "trajectory_repair":
                    # A-EQUAL is already converged.  Update only the mined
                    # off-policy boundary plus safety constraints and KL.
                    loss = chosen_ce * 0.0
                if args.method in {
                    "short_stop", "equal_cov_stop", "equal_full_stop",
                    "redundancy_stop", "cequal_dual", "aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl",
                    "trajectory_equal",
                }:
                    loss = loss + args.stop_alpha * stop_nll
                if args.method == "length_pref":
                    rejected = row["long"]
                    rejected_lp, _, _ = response_logp(model, tokenizer, row["prompt"], rejected, args.max_length)
                    loss = loss + args.pref_alpha * -F.logsigmoid(args.beta * (chosen_lp - rejected_lp))
                elif args.method == "equal_rag":
                    key = random.choice(["negative_drop", "negative_swap", "negative_redundant"])
                    rejected_lp, _, _ = response_logp(model, tokenizer, row["prompt"], row[key], args.max_length)
                    loss = loss + args.pref_alpha * -F.logsigmoid(args.beta * (chosen_lp - rejected_lp))
                elif args.method in {"equal_cov", "equal_cov_stop", "equal_full_stop"}:
                    negative_keys = ["negative_drop", "negative_swap"]
                    if args.method == "equal_full_stop":
                        negative_keys.append("negative_redundant")
                    pref_losses = []
                    for key in negative_keys:
                        rejected_lp, _, _ = response_logp(
                            model, tokenizer, row["prompt"], row[key], args.max_length
                        )
                        pref_losses.append(-F.logsigmoid(args.beta * (chosen_lp - rejected_lp)))
                    loss = loss + args.pref_alpha * torch.stack(pref_losses).mean()
                elif args.method == "redundancy_stop":
                    rejected_lp, _, _ = response_logp(
                        model, tokenizer, row["prompt"], row["negative_redundant"], args.max_length
                    )
                    loss = loss + args.pref_alpha * -F.logsigmoid(
                        args.beta * (chosen_lp - rejected_lp)
                    )
                elif args.method in {"cequal_dual", "aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"}:
                    if args.method == "uequal_kl":
                        unit_prompt, unit_chosen, unit_drop, unit_swap, unit_redundant = (
                            evidence_unit_variants(row)
                        )
                        preference_chosen_lp, _, _ = response_logp(
                            model, tokenizer, unit_prompt, unit_chosen, args.max_length
                        )
                        redundant_lp, _, _ = response_logp(
                            model, tokenizer, unit_prompt, unit_redundant, args.max_length
                        )
                        negative_specs = ((unit_prompt, unit_drop), (unit_prompt, unit_swap))
                    else:
                        preference_chosen_lp = chosen_lp
                        redundant_lp, _, _ = response_logp(
                            model, tokenizer, row["prompt"], row["negative_redundant"], args.max_length
                        )
                        negative_specs = tuple(
                            (row["prompt"], row[key]) for key in ("negative_drop", "negative_swap")
                        )
                    compression_loss = -F.logsigmoid(
                        args.beta * (preference_chosen_lp - redundant_lp)
                    )
                    constraint_terms = []
                    for negative_prompt, negative_response in negative_specs:
                        rejected_lp, _, _ = response_logp(
                            model, tokenizer, negative_prompt, negative_response, args.max_length
                        )
                        gap = preference_chosen_lp - rejected_lp
                        constraint_terms.append(F.relu(args.constraint_margin - gap))
                    constraint_violation = torch.stack(constraint_terms).mean()
                    loss = (
                        loss
                        + args.pref_alpha * compression_loss
                        + dual_value * constraint_violation
                    )
                    saturation_loss = None
                    trajectory_loss = None
                    if args.method == "sequal_kl":
                        saturation_loss, saturation_complete, saturation_incomplete = (
                            saturation_boundary_loss(
                                model, tokenizer, row, args.max_length,
                                args.saturation_margin,
                            )
                        )
                        loss = loss + args.saturation_alpha * saturation_loss
                    if args.method in {"trajectory_equal", "trajectory_repair"} and row.get("has_trajectory", False):
                        trajectory_loss, trajectory_end_lp, trajectory_continuation_lp, trajectory_max_lp = (
                            trajectory_boundary_loss(
                                model, tokenizer, row, args.max_length,
                                args.trajectory_margin,
                            )
                        )
                        loss = loss + args.trajectory_alpha * trajectory_loss
                    anchor_violation = None
                    endpoint_violation = None
                    if args.method in {"aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"}:
                        if args.method in {"aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"} and args.anchor_logits_dir:
                            cache_file = anchor_cache_path(args.anchor_logits_dir, row["id"])
                            if not cache_file.exists():
                                raise FileNotFoundError(f"Missing cached anchor logits: {cache_file}")
                            anchor_logits = torch.load(
                                cache_file, map_location=chosen_logits.device, weights_only=True
                            )
                            if tuple(anchor_logits.shape) != tuple(chosen_logits.shape):
                                raise ValueError(
                                    f"Cached anchor shape {tuple(anchor_logits.shape)} != "
                                    f"student shape {tuple(chosen_logits.shape)} for {row['id']}"
                                )
                        else:
                            if anchor_model is None:
                                raise ValueError(
                                    f"{args.method} requires --anchor-adapter or --anchor-logits-dir"
                                )
                            with torch.no_grad():
                                if args.method in {"aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"}:
                                    anchor_lp, _, _, anchor_logits = response_logp(
                                        anchor_model, tokenizer, row["prompt"], row["short"],
                                        args.max_length, return_response_logits=True,
                                    )
                                else:
                                    anchor_lp, _, _ = response_logp(
                                        anchor_model, tokenizer, row["prompt"], row["short"], args.max_length
                                    )
                        if args.method in {"aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"}:
                            anchor_logits = anchor_logits.to(chosen_logits.device)
                            anchor_log_probs = F.log_softmax(anchor_logits.float(), dim=-1)
                            chosen_log_probs = F.log_softmax(chosen_logits.float(), dim=-1)
                            anchor_kl = (
                                anchor_log_probs.exp() * (anchor_log_probs - chosen_log_probs)
                            ).sum(dim=-1).mean()
                            anchor_violation = F.relu(anchor_kl - args.anchor_kl_target)
                            if args.endpoint_alpha > 0:
                                # Sequence-averaged KL can hide a large change at the
                                # single clean stopping position.  Preserve the frozen
                                # A-EQUAL EOS-vs-best-continuation margin explicitly.
                                eos_id = tokenizer.eos_token_id
                                anchor_end = anchor_log_probs[-1]
                                chosen_end = chosen_log_probs[-1]
                                anchor_non_end = anchor_end.clone()
                                chosen_non_end = chosen_end.clone()
                                anchor_non_end[eos_id] = -torch.inf
                                chosen_non_end[eos_id] = -torch.inf
                                anchor_endpoint_margin = anchor_end[eos_id] - anchor_non_end.max()
                                chosen_endpoint_margin = chosen_end[eos_id] - chosen_non_end.max()
                                endpoint_violation = F.relu(
                                    torch.abs(chosen_endpoint_margin - anchor_endpoint_margin)
                                    - args.endpoint_margin_slack
                                )
                        else:
                            # Absolute sequence-probability trust region.
                            anchor_violation = F.relu(
                                anchor_lp - args.anchor_slack - chosen_lp
                            )
                        loss = loss + args.anchor_alpha * anchor_violation
                        if endpoint_violation is not None:
                            loss = loss + args.endpoint_alpha * endpoint_violation
                    pending_violations.append(float(constraint_violation.detach().cpu()))
            else:
                raise ValueError(args.method)
            (loss / args.grad_accum).backward()
            step += 1
            if step % args.grad_accum == 0 or step == len(loader) * args.epochs:
                last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); optim.zero_grad(set_to_none=True)
                if args.method in {"cequal_dual", "aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"} and pending_violations:
                    signal = sum(pending_violations) / len(pending_violations)
                    dual_ema = args.dual_ema_beta * dual_ema + (1 - args.dual_ema_beta) * signal
                    dual_value = min(
                        args.dual_max,
                        max(0.0, dual_value + args.dual_lr * (dual_ema - args.constraint_target)),
                    )
                    pending_violations.clear()
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "row_id": row.get("id"),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": (
                        float(last_grad_norm.detach().cpu())
                        if last_grad_norm is not None else None
                    ),
                }
                if args.method in {"cequal_dual", "aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"}:
                    record |= {
                        "dual_value": dual_value,
                        "constraint_ema": dual_ema,
                    }
                    if args.method in {"aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"} and anchor_violation is not None:
                        record["anchor_violation"] = float(anchor_violation.detach().cpu())
                        if args.method in {"aequal_kl", "uequal_kl", "sequal_kl", "trajectory_equal", "trajectory_repair"}:
                            record["anchor_kl"] = float(anchor_kl.detach().cpu())
                            if endpoint_violation is not None:
                                record["endpoint_violation"] = float(endpoint_violation.detach().cpu())
                                record["anchor_endpoint_margin"] = float(anchor_endpoint_margin.detach().cpu())
                                record["chosen_endpoint_margin"] = float(chosen_endpoint_margin.detach().cpu())
                    if args.method == "sequal_kl" and saturation_loss is not None:
                        record |= {
                            "saturation_loss": float(saturation_loss.detach().cpu()),
                            "saturation_complete": float(saturation_complete.detach().cpu()),
                            "saturation_incomplete": float(saturation_incomplete.detach().cpu()),
                        }
                    if args.method in {"trajectory_equal", "trajectory_repair"}:
                        record["trajectory_active"] = trajectory_loss is not None
                        if trajectory_loss is not None:
                            record |= {
                                "trajectory_prefix_tokens": len(tokenizer(
                                    row["trajectory_prefix"], add_special_tokens=False
                                )["input_ids"]),
                                "trajectory_loss": float(trajectory_loss.detach().cpu()),
                                "trajectory_end_lp": float(trajectory_end_lp.detach().cpu()),
                                "trajectory_continuation_lp": float(trajectory_continuation_lp.detach().cpu()),
                                "trajectory_max_lp": float(trajectory_max_lp.detach().cpu()),
                            }
                history.append(record); print(json.dumps(record), flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    return history


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\[\d+\]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def contains_alias(text: str, aliases: list[str]) -> bool:
    norm = f" {normalize(text)} "
    return any(f" {normalize(a)} " in norm for a in aliases if normalize(a))


@torch.inference_mode()
def evaluate(args, model, tokenizer, rows, out_dir: Path):
    model.eval(); model.config.use_cache = True
    device = next(model.parameters()).device
    results = []
    for idx, row in enumerate(rows):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not args.disable_thinking,
        )
        enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen_ids = out[0, enc["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        found, grounded, cited = 0, 0, 0
        for edge in row["edges"]:
            if contains_alias(text, edge["aliases"]):
                found += 1
                # Inspect a compact window following an alias for the citation.
                locs = [text.lower().find(a.lower()) for a in edge["aliases"] if a.lower() in text.lower()]
                locs = [x for x in locs if x >= 0]
                if locs:
                    window = text[min(locs): min(locs) + 160]
                    refs = {int(x) for x in re.findall(r"\[(\d+)\]", window)}
                    if refs: cited += 1
                    if edge["doc_id"] in refs: grounded += 1
        all_refs = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
        valid_refs = sum(1 for x in all_refs if 1 <= x <= len(row["docs"]))
        results.append({
            "id": row["id"], "output": text,
            "tokens": int(gen_ids.numel()), "gold_edges": len(row["edges"]),
            "answers_found": found, "answers_cited": cited, "grounded_edges": grounded,
            "citations": len(all_refs), "valid_citations": valid_refs,
        })
        if (idx + 1) % 20 == 0: print(f"eval {idx+1}/{len(rows)}", flush=True)
    n_edges = sum(x["gold_edges"] for x in results)
    metrics = {
        "n": len(results),
        "answer_recall": sum(x["answers_found"] for x in results) / max(n_edges, 1),
        "citation_coverage": sum(x["answers_cited"] for x in results) / max(n_edges, 1),
        "grounded_edge_recall": sum(x["grounded_edges"] for x in results) / max(n_edges, 1),
        "citation_validity": sum(x["valid_citations"] for x in results) / max(sum(x["citations"] for x in results), 1),
        "mean_output_tokens": sum(x["tokens"] for x in results) / len(results),
    }
    metrics["grounded_utility_per_token"] = metrics["grounded_edge_recall"] / max(metrics["mean_output_tokens"], 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval_results.jsonl").open("w", encoding="utf-8") as f:
        for x in results: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--method",
        choices=[
            "long_sft", "short_sft", "length_pref", "equal_rag",
            "short_stop", "equal_cov", "equal_cov_stop", "equal_full_stop", "redundancy_stop",
            "cequal_dual", "aequal_dual", "aequal_kl", "uequal_kl", "sequal_kl",
            "trajectory_equal", "trajectory_repair",
        ],
        required=True,
    )
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--pref-alpha", type=float, default=0.5)
    ap.add_argument("--stop-alpha", type=float, default=2.0)
    ap.add_argument("--saturation-alpha", type=float, default=1.0)
    ap.add_argument("--saturation-margin", type=float, default=1.0)
    ap.add_argument("--trajectory-alpha", type=float, default=0.2)
    ap.add_argument("--trajectory-margin", type=float, default=1.0)
    ap.add_argument("--constraint-margin", type=float, default=0.1)
    ap.add_argument("--constraint-target", type=float, default=0.02)
    ap.add_argument("--dual-init", type=float, default=0.5)
    ap.add_argument("--dual-lr", type=float, default=0.5)
    ap.add_argument("--dual-max", type=float, default=5.0)
    ap.add_argument("--dual-ema-beta", type=float, default=0.9)
    ap.add_argument("--init-adapter", default="", help="PEFT adapter used to initialize trainable weights")
    ap.add_argument("--anchor-adapter", default="", help="Frozen PEFT adapter for the absolute quality anchor")
    ap.add_argument(
        "--anchor-logits-dir", default="",
        help="Per-example frozen anchor logits cache for aequal_kl; avoids a second online model",
    )
    ap.add_argument("--anchor-alpha", type=float, default=2.0)
    ap.add_argument("--anchor-slack", type=float, default=0.02)
    ap.add_argument("--anchor-kl-target", type=float, default=0.01)
    ap.add_argument("--endpoint-alpha", type=float, default=0.0)
    ap.add_argument("--endpoint-margin-slack", type=float, default=0.0)
    ap.add_argument("--anchor-device", default="cuda", help="Device for frozen anchor, e.g. cuda:1")
    ap.add_argument("--disable-thinking", action="store_true", help="Disable Qwen3 thinking mode in chat templates")
    ap.add_argument(
        "--disable-lora-dropout",
        action="store_true",
        help="Use deterministic LoRA branches so student/anchor KL is comparable",
    )
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="Diagnostic cap on training steps")
    ap.add_argument("--adapter", default="", help="Existing PEFT adapter for evaluation-only mode")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    seed_all(args.seed)
    train_rows, test_rows = read_jsonl(args.train), read_jsonl(args.test)
    if args.max_train: train_rows = train_rows[: args.max_train]
    if args.max_test: test_rows = test_rows[: args.max_test]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, use_fast=True)
    tokenizer._equal_rag_disable_thinking = args.disable_thinking
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    # Assemble PEFT adapters on CPU before moving the complete model.  Loading
    # an adapter after its base has already been placed on different CUDA
    # devices can leave separately loaded copies with inconsistent LoRA state.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if args.eval_only:
        if not args.adapter:
            raise ValueError("--eval-only requires --adapter")
        model = PeftModel.from_pretrained(model, args.adapter).cuda()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = vars(args) | {"train_examples": 0, "test_examples": len(test_rows)}
        (out_dir / "eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        evaluate(args, model, tokenizer, test_rows, out_dir)
        return
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                            task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    if args.disable_lora_dropout:
        disabled = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout) and module.p:
                module.p = 0.0
                disabled += 1
        print(f"disabled dropout in {disabled} modules", flush=True)
    model = model.cuda()
    anchor_model = None
    if args.anchor_adapter:
        anchor_base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16
        )
        anchor_model = PeftModel.from_pretrained(
            anchor_base, args.anchor_adapter
        ).to(args.anchor_device)
        anchor_model.eval()
        anchor_model.config.use_cache = False
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)
    model.print_trainable_parameters()
    started = time.time()
    history = train(args, model, tokenizer, train_rows, anchor_model=anchor_model)
    out_dir = Path(args.out_dir)
    manifest = vars(args) | {"train_examples": len(train_rows), "test_examples": len(test_rows),
                             "elapsed_train_seconds": time.time() - started, "history": history}
    save_adapter(model, tokenizer, out_dir, manifest)
    evaluate(args, model, tokenizer, test_rows, out_dir)


if __name__ == "__main__":
    main()

"""Verified semantic-EOT repair and matched controls, leaving legacy code intact.

UL is an answer-unit adaptation of sequence unlikelihood, NOT a full DITTO
reproduction. All arms reuse frozen on-policy trajectories and cached anchor KL.
"""
import hashlib
import json
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GenerationConfig
import train_trajectory_equal as legacy

ARM = os.environ.get('EQUAL_VERIFIED_ARM', 'semantic_eos')
assert ARM in {'semantic_eos', 'unit_ul', 'wrong_boundary', 'kl_only'}


@lru_cache(maxsize=8)
def model_eos_ids(path):
    value = GenerationConfig.from_pretrained(path).eos_token_id
    return [value] if isinstance(value, int) else list(value)


def eos_ids(tokenizer):
    return model_eos_ids(tokenizer.name_or_path)


def boundary_ids(tokenizer, prompt, response, max_length):
    ids, mask, labels = legacy.encode_example(tokenizer, prompt, response, max_length)
    stops = eos_ids(tokenizer)
    positions = [int(p) for p in labels.ne(-100).nonzero().flatten()
                 if int(ids[p]) in stops]
    if not positions:
        raise ValueError('No assistant EOT found: response/template may be truncated')
    pos = positions[0]
    if pos == 0:
        raise ValueError('Invalid EOT position')
    # Remove both EOT and anything following it; never score a post-EOT newline.
    return ids[:pos], mask[:pos], int(ids[pos]), stops


def boundary_distribution(model, tokenizer, row, max_length, wrong=False):
    response = row['trajectory_prefix']
    if wrong:
        # Deterministic earlier *answer-unit* boundary, not a new failure screen.
        units = response.split(', ')
        if len(units) >= 2:
            index = int(hashlib.sha256(str(row['id']).encode()).hexdigest(), 16) % (len(units)-1) + 1
            response = ', '.join(units[:index])
        else:
            # One-item trajectories have no earlier unit boundary; use the
            # start of the assistant answer, recorded as a negative control.
            response = ''
    if wrong:
        ids, mask, stop, stops = boundary_ids(tokenizer, row['prompt'], response, max_length)
    else:
        ids, _, _, stop, stops, _ = aligned_trajectory(tokenizer, row, max_length)
        mask = torch.ones_like(ids)
    device = next(model.parameters()).device
    logits = model(input_ids=ids[None].to(device), attention_mask=mask[None].to(device)).logits[0, -1].float()
    lp = F.log_softmax(logits, -1)
    non_stop = lp.clone()
    non_stop[stops] = -torch.inf
    return lp[stop], non_stop.max(), stop, int(ids.numel())


def aligned_trajectory(tokenizer, row, max_length):
    response = row['trajectory_prefix'] + row['trajectory_continuation']
    full = tokenizer.apply_chat_template(
        [{'role':'user','content':row['prompt']},{'role':'assistant','content':response}],
        tokenize=False, add_generation_prompt=False,
        enable_thinking=not getattr(tokenizer,'_equal_rag_disable_thinking',False))
    start = full.rfind(response)
    if start < 0:
        raise ValueError('Cannot locate response in chat template')
    char_boundary = start + len(row['trajectory_prefix'])
    enc = tokenizer(full, return_offsets_mapping=True, truncation=False)
    ids = torch.tensor(enc['input_ids'],dtype=torch.long)
    stops = eos_ids(tokenizer)
    end = next(i for i,(token,offset) in enumerate(zip(ids.tolist(),enc['offset_mapping']))
               if token in stops and offset[0] >= start)
    # If BPE merges `],`, retain the full token (closing citation + separator).
    # Stop at the first token boundary before the duplicate answer's content.
    n = next(i+1 for i,(lo,hi) in enumerate(enc['offset_mapping']) if hi >= char_boundary and hi > lo)
    if not 0 < n < end < max_length:
        raise ValueError('Truncated or empty repetition continuation')
    crossed = enc['offset_mapping'][n-1][1]-char_boundary
    return ids[:n], ids[:end], n, int(ids[end]), stops, crossed


def unit_unlikelihood(model, tokenizer, row, max_length):
    context, combined, n, _, _, _ = aligned_trajectory(tokenizer, row, max_length)
    device = next(model.parameters()).device
    logits = model(input_ids=combined[None].to(device)).logits[0, n-1:-1].float()
    targets = combined[n:].to(device)
    lp = F.log_softmax(logits, -1).gather(-1, targets[:, None]).squeeze(-1)
    return -torch.log1p(-lp.exp().clamp(max=1-1e-6)).mean()


def train(args, model, tokenizer, rows, anchor_model=None):
    if args.method != 'trajectory_repair' or args.pref_alpha or args.stop_alpha or args.dual_max:
        raise ValueError('This matched study only supports frozen repair+KL configuration')
    loader = DataLoader(legacy.Rows(rows), batch_size=1, shuffle=True, collate_fn=legacy.one)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    model.config.use_cache = False
    opt.zero_grad(set_to_none=True)
    history, active, positive, count, last_grad = [], 0, 0, 0, None
    for epoch in range(args.epochs):
        for row in loader:
            _, _, _, student = legacy.response_logp(model, tokenizer, row['prompt'], row['short'],
                                                    args.max_length, return_response_logits=True)
            anchor = torch.load(legacy.anchor_cache_path(args.anchor_logits_dir, row['id']),
                                map_location=student.device, weights_only=True)
            if anchor.shape != student.shape:
                raise ValueError(f"Anchor shape mismatch for {row['id']}")
            a = F.log_softmax(anchor.float(), -1)
            s = F.log_softmax(student.float(), -1)
            kl = (a.exp() * (a-s)).sum(-1).mean()
            safety = args.anchor_alpha * F.relu(kl-args.anchor_kl_target)
            kl_value = float(kl.detach())
            loss_value = float(safety.detach())
            (safety / args.grad_accum).backward()
            del student, anchor, a, s, safety, kl
            boundary = None
            record = {'row_id': row['id'], 'arm': ARM, 'anchor_kl': kl_value}
            if row.get('has_trajectory') and ARM != 'kl_only':
                active += 1
                if ARM == 'unit_ul':
                    boundary = unit_unlikelihood(model, tokenizer, row, args.max_length)
                else:
                    end, cont, stop, nctx = boundary_distribution(model, tokenizer, row,
                                                                 args.max_length, ARM == 'wrong_boundary')
                    boundary = F.relu(args.trajectory_margin-(end-cont))
                    record.update(eot_id=stop, context_tokens=nctx, end_lp=float(end.detach()),
                                  best_cont_lp=float(cont.detach()))
                value = float(boundary.detach())
                positive += int(value > 0)
                loss_value += args.trajectory_alpha * value
                record['boundary_loss'] = value
                (args.trajectory_alpha * boundary / args.grad_accum).backward()
                del boundary
            count += 1
            if count % args.grad_accum == 0 or count == len(loader)*args.epochs or (args.max_steps and count >= args.max_steps):
                last_grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not torch.isfinite(torch.tensor(last_grad)):
                    raise ValueError('Non-finite gradient')
                opt.step()
                opt.zero_grad(set_to_none=True)
            if count == 1 or count % args.log_every == 0 or count == len(loader)*args.epochs:
                record.update(step=count, loss=loss_value, gradient_norm=last_grad,
                              active_boundaries=active, positive_boundary_losses=positive,
                              peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                history.append(record)
                print(json.dumps(record), flush=True)
            if args.max_steps and count >= args.max_steps:
                break
        if args.max_steps and count >= args.max_steps:
            break
    audit = {'arm': ARM, 'seen_rows': count, 'active_boundaries': active,
             'positive_boundary_losses': positive, 'semantic_eot_ids': eos_ids(tokenizer),
             'legacy_post_eot_suffix_excluded': True,
             'anchor_kl_template_unchanged_for_cache_compatibility': True,
             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'boundary_training_audit.json').write_text(json.dumps(audit, indent=2))
    return history


legacy.train = train
if __name__ == '__main__':
    legacy.main()

# Reproduction: public statistics vs owner-asset GPU replay

## CPU quick start

```bash
python -m pip install -r requirements-analysis.txt
python scripts/show_results.py
python scripts/verify_release.py
python scripts/run_checks.py
python scripts/reproduce_supplementary.py
```

Python3.10+, NumPy1.26–2.x. Original independent Windows checking used NumPy2.2.6. Viewer/verifier use only stdlib. The last command rebuilds325 points,65 ledgers and1425 delta/CI values from anonymous per-question six-column arrays: grounded groups, supported groups, tokens, Full F1, repetition flag, true-cap flag. Original lexicographic sampling order is preserved; exact archived statistical functions use20000 paired question draws, RNG20261010, undefined draws excluded only from Q. No account, GPU or model download is needed.

This validates published arithmetic/uncertainty, not generation, text matching or semantic support. Main three-Seed/hierarchical and development ablation aggregates are also exported, but this replay command specifically reconstructs supplementary single-Seed S2/S3.

## Frozen GPU settings

Seed20261015; LoRA r8/alpha16/dropout0 on q/k/v/o/gate/up/down projections; AdamW lr2e-5, accumulation4, clipping1; exactly60 steps/240 supervised visits. Whole-answer SFT covers the entire positive response and EOS. Sequence DPO uses frozen Base, summed response log probabilities, β0.1 and0.1 whole-answer CE. Identical C2 pairs receive no artificial preference gradient. EQUAL uses positive repairs plus10KL+2 changed-only margin0.5 and240 auxiliary retention visits; AOnly action β0.25 and citation β/2, no C1/C2 negative Hinge. Matched steps/positive visits are not equal FLOPs or controlled wall time.

All final inference uses unmerged LoRA, greedy decoding, thinking disabled, cap256. No main-method rules; Entity Blocker alone has a decoder rule. Four frozen Qwen controls exactly reproduced380/380 original token paths.

Qwen uses the official unadapted Base, not old Anchor initialization. Mistral revision: `c170c708c41dac9275d15a8fff4eca08d52bab71`; [model/tokenizer file hashes](../results/supplementary_20261017/package3/model_identity.json). Qwen pool SHA256: `0d91e2e47fc2089183ea3cfa7e45ba9ae79e78de9fd26c95cfbc18dcbba85ceb`,21A+20C2 with21 embeddedC1. Mistral has15A+10C2/15 embeddedC1;120 retention trajectories/5317 positions. Same updates, different native pool sizes/effective epochs.

## Required assets and configurable paths

Current dated source: [research snapshot](../research/trajectory_repair_20261017/). Exact GPU replay needs separately obtained original questions/passages, tokenizer-dependent compiled labels, native trajectories, archived main adapters and reference logits. Numeric result arrays cannot recreate those assets. Never delete frozen hash asserts to force another dataset through historical code.

```bash
export EQUAL_RAG_ROOT=/path/to/owner-prepared-workspace
export EQUAL_RAG_QWEN_MODEL=/path/to/Qwen3-14B
export EQUAL_RAG_MISTRAL_MODEL=/path/to/Mistral-7B-Instruct-v0.3
export EQUAL_RAG_CACHE_ROOT=/path/to/local-cache
```

The workspace needs the original named `runs/` and, for some audits, `artifacts/` snapshots. CUDA/Linux, PyTorch, Transformers, PEFT and tokenizer dependencies are needed; see training requirements and manifests. This is **not a turnkey SDK**. Final Qwen loading uses `run_qwen_single_v3.py`; directly running `run_matched_baselines.py` retains the historical multi-device loader. Mistral compiler/training is v2; v3 only shards independent preparation questions at batch size1 and restores order.

Historical preparation/launch/gate/merge functions remain in dated dependencies for audit, not instructions to execute every `main()`. Final supplement has no new training gate/checkpoint selection. `release_paths.py` rewrites private paths only; original/exported hashes record this distribution transformation. The exact statistical function bodies are mechanically extracted unchanged and AST-compared in CPU tests.

Raw text/real-answer token traces/weights/logits/logs are not bundled. Literal and synthetic tokenizer examples plus21 mismatch decisions are public; redoing real21-record tokenization needs original repair records. For old isolated-site experiments see [the dated2026-10-11 guide](REPRODUCIBILITY_20261011.md). The supplement is post-unsealing, not another blind study.

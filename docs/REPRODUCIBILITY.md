# Reproduction levels and commands

## Level 1: inspect the released evidence

```bash
python scripts/show_results.py
python scripts/verify_release.py
```

Uses Python standard library only. The viewer reads saved aggregate JSON; no fabricated or regenerated metrics. The verifier checks source/result hashes, Python syntax, internal documentation links, prohibited artifact types, and common private-path/credential patterns. It does not establish complete absence of secrets or research correctness.

## Level 2: CPU synthetic checks

```bash
python -m pip install -r requirements-analysis.txt
python scripts/run_checks.py
```

Checks strict unit-local scoring and completion-repair invariants using synthetic labels. The historical data-dependent test in `test_completion_repair_20261010.py` is not part of this lightweight suite because benchmark/prepared rows are deliberately not distributed. GPU/tokenizer-dependent historical suites are likewise not represented as passing by the CPU checks.

## Level 3: training-code inspection / historical replay

Observed completed runtime: Linux, Python3.10.20, PyTorch2.7.1/CUDA12.6, Transformers4.57.3, PEFT0.18.0, A100-SXM4-40GB. Latest three arms used separate GPUs, one GPU per training arm. Training uses BF16 and batch1/gradient accumulation8.

`requirements-training.txt` records the main stack, but is **not a complete transitive lockfile**. Choose the platform-appropriate CUDA wheel; we have not rerun the full14B training on the Windows publication machine. Exact replay also needs the original anchor adapter, prepared train/smoke labels, cached teacher logits, official alias groups and paired prediction files. These are excluded assets, not quietly supplied by the quick start.

Paths are configurable through `research/release_paths.py`:

```bash
export EQUAL_RAG_ROOT=/path/to/experiment-workspace
export EQUAL_RAG_QWEN_MODEL=/path/to/models/Qwen3-14B
export EQUAL_RAG_CACHE_ROOT=/path/to/anchor-caches
export EQUAL_QWEN_TERMINAL_ARM=action_pair  # or action_ce / action_ul
```

Use an isolated new workspace, never a completed original run directory. Expected latest prepared-data layout:

```text
experiment-workspace/
├── runs/qwen_native_consistent_20261010/compiled/{train.jsonl,audit.json,smoke.jsonl}
├── runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter/
├── data/qampari_evidence_full/dev.jsonl
├── data/prospective_qampari_gtr5_v2_20261010/dev.jsonl
└── ALCE/data/qampari_eval_gtr_top100.json
anchor-caches/qwen_retrieval_align_20261010/*.pt
```

If you have the authorized prepared assets and main model-stack dependencies, this is the historical training command pattern (not run automatically by the release):

```bash
CUDA_VISIBLE_DEVICES=0 python research/train_qwen_native_consistent_20261010.py \
  --model "$EQUAL_RAG_QWEN_MODEL" \
  --train "$EQUAL_RAG_ROOT/runs/qwen_native_consistent_20261010/compiled/train.jsonl" \
  --test "$EQUAL_RAG_ROOT/data/qampari_evidence_full/dev.jsonl" \
  --out-dir "$EQUAL_RAG_ROOT/runs/replay_action_pair/train" \
  --method trajectory_repair --seed 20260930 --epochs 1 --lr 1e-5 \
  --grad-accum 8 --max-length 2048 --max-new-tokens 256 \
  --init-adapter "$EQUAL_RAG_ROOT/runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter" \
  --anchor-logits-dir "$EQUAL_RAG_CACHE_ROOT/qwen_retrieval_align_20261010" \
  --anchor-alpha 10 --anchor-kl-target 0.0001 \
  --trajectory-alpha 0.1 --trajectory-margin 1 \
  --stop-alpha 0 --pref-alpha 0 --dual-init 0 --dual-lr 0 --dual-max 0 \
  --disable-lora-dropout --disable-thinking --log-every 20
```

The wrapper deliberately skips its legacy evaluation; evaluation is separate. It does not execute a new coherent-span method. Prepared-data asserts are historical validity checks, not obstacles to remove.

Interpolate then evaluate with exact telemetry:

```bash
python research/interpolate_peft_adapters.py \
  --base "$EQUAL_RAG_ROOT/runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter" \
  --tuned "$EQUAL_RAG_ROOT/runs/replay_action_pair/train/adapter" \
  --output "$EQUAL_RAG_ROOT/runs/replay_action_pair/interp05/adapter" --weight 0.5

CUDA_VISIBLE_DEVICES=0 python research/eval_verified_outputs_20261010.py \
  --model "$EQUAL_RAG_QWEN_MODEL" \
  --train "$EQUAL_RAG_ROOT/runs/qwen_native_consistent_20261010/compiled/train.jsonl" \
  --test "$EQUAL_RAG_ROOT/data/qampari_evidence_full/dev.jsonl" \
  --out-dir "$EQUAL_RAG_ROOT/runs/replay_action_pair/regression_dev" \
  --method trajectory_repair --seed 20260930 --eval-only \
  --adapter "$EQUAL_RAG_ROOT/runs/replay_action_pair/interp05/adapter" \
  --disable-thinking --max-length 2048 --max-new-tokens 256
```

Raw-retrieval evaluation uses its own data path; never label the two different question sets a same-question retrieval ablation. Record fresh manifests and do not merge new metrics with the historical frozen summary.

## Exact replay limitations

- The source hashes in original training manifests describe original files. Release files have sanitized paths and their own hashes.
- Compilers/finalizers require historical predecessor artifacts and assert expected source hashes. Some finalizers expect source copies under `workspace/code`; that is a historical layout, not the release `research/` location.
- Aggregate JSON alone cannot recreate confidence intervals; question-level paired records are excluded.
- CPU checks do not validate GPU numerics, BF16 training, adapter equivalence, or the NLI judge's real-data calibration.
- The latest version is one training seed and reused development. Do not treat reproduction of that development score as independent confirmation.

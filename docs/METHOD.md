# Current method: native-tail counterfactual repair

Current through 2026-10-17. The [2026-10-11 method](METHOD_20261011.md) is historical. Its isolated two-site loss and old Anchor initialization are **not** the current framework.

## Evidence states

`E_sup(D)` is the normalized known answer groups lexically supported in documents; `E_past` is the groups already emitted; `ΔE = E_sup − E_past`. These training/evaluation proxies are not semantic completeness certificates. Inference receives no gold labels or ΔE.

| State | Allowed location | Positive target | Restriction |
|---|---|---|---|
| A: new-entity completion | Native Base tail with remaining known support | Genuinely new entity + supporting citation, then closure | Adding a second citation to an old name is not a new entity |
| C1: completion closure | End of A repair | Closure + EOS | Embedded in A, not another independent record |
| C2: stop retention | Native exhausted stop under known inventory | Preserve native closure/EOS | Never truncate ordinary nonrepeating mid-list continuation |

Qwen has **21 A + 20 C2 = 41 records**, with 21 embedded C1 sites. Mistral native compilation has **15 A + 10 C2 = 25 records**, with 15 embedded C1 sites. These final pools contain no B records; historical bridge proposals are not silently counted as current evidence.

## Boundary alignment

Separately encoding text appended after `]` can differ from jointly encoding `], ...`. The compiler rolls back a small native tail, jointly re-tokenizes, then finds the first positive/negative token divergence. Shared prefix tokens remain context, not contradictory labels.

The audit's 21/21 mismatch rates demonstrate segmentation inconsistency, not zero continuation probability or universal hard deadlock. Standard whole-answer SFT already encodes full targets jointly. Standalone IDs may differ from contextual IDs, particularly in SentencePiece; use real contexts.

## Current objective

Exact implementation: [current source](../research/trajectory_repair_20261017/). These are dated research programs, not a generic training SDK.

```text
positive supervision:
    first-action CE + mean repair-span CE
    + 2 × C1 terminal CE + first-new-entity semantic CE (A only)

prefix retention R:
    10 × selective KL(Base || student)
    + 2 × changed-top1-only mean relu(0.5 − margin)

optional A-Only contrast:
    β × action Hinge + (β/2) × citation Hinge, margin 2
    no C1/C2 negative continuation Hinge

auxiliary retention:
    0.5 × R on one additional native retention visit per supervised visit
```

Allowed closure/EOS classes use total probability mass. C2 has no A semantic/C1 term. KL target is 0; retention is soft and changed-top-1-only. No Max-Guard is used in final configurations. β=0 is **CE repair**, not renamed Pair.

Implementation pointers in the current source folder:

- `qwen_tail_fidelity_v3_20261015.py`: positive first/span/terminal/semantic loss.
- `qwen_tail_fidelity_v2_20261015.py`: mean changed-top-1 retention.
- `qwen_pair_aonly_20261015.py`: disable C1/C2 Hinge, fixed-step A-only variants.
- `entity_abc_contracts_20261015.py`: normalized inventory and entity contracts.
- `run_matched_baselines.py`: full-answer CE and sum-log-prob DPO + 0.1 SFT.
- `run_second_backbone_v2.py`: native Mistral labels, loss and diagnostics.
- `single_gpu_runtime_v3.py`: isolated-A100 Qwen loading, no static merging.

## Optimization and inference

Original Base is both initialization and KL reference. LoRA r8/alpha16/dropout0 on q/k/v/o/gate/up/down projections; AdamW lr2e-5, accumulation4, clipping1, exactly60 updates/240 supervised visits, Seed20261015 in the supplement. EQUAL adds240 auxiliary retention visits: **not matched FLOPs**.

Inference is unmerged LoRA, ordinary greedy generation, no thinking, cap256. BF16 static merging and Max-Guard are diagnostic variants, not current methods. Observed static-merge degradation is not proof that every BF16 merge fails.

## Evidence boundaries

Development three-Seed/12-ablation studies, the frozen three-Seed Full311 evaluation, and later post-unsealing baselines/Mistral are separate studies. Mistral AOnly improves over Base; pure CE transfer intervals cross zero. New Qwen baselines have higher Q. See [evidence ledger](EXPERIMENTS.md).

Teacher-forced Ret./120 is not free-generation preservation on unseen questions, factual correctness or guaranteed no answer loss. No universal model/tokenizer behavior, semantic-entailment improvement or controlled serving acceleration is established.

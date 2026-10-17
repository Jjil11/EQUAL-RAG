# EQUAL-RAG

Evidence-aware trajectory repair for multi-answer, citation-grounded generation.

[简体中文](README.zh-CN.md) · [Method](docs/METHOD.md) · [All experiments](docs/EXPERIMENTS.md) · [Reproduce statistics](docs/REPRODUCIBILITY.md) · [Supplementary report](results/supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md)

> Updated through the **2026-10-17 supplementary experiments**. Research evidence, including negative results—not an accepted-paper release or a claim of superiority over strong baselines.
>
> **Important correction:** EQUAL improves over the unadapted Base, but the new exhaustive-prompt and budget-matched SFT/DPO baselines achieve higher Q on Qwen3-14B Full311. The old inflated Anchor is insufficient evidence of an algorithmic advantage.

## What the method does

EQUAL-RAG repairs the model's native answer tail rather than routing decoding at inference. Its current framework uses:

1. **A/C1/C2 native-tail labels:** add a genuinely missing supported entity; close after completion; retain a native exhausted stop.
2. **Boundary rollback and joint re-tokenization:** preserve a valid continuation token sequence across a local repair boundary.
3. **Prefix retention:** selective KL plus a changed-top-1 margin penalty, referenced to the original Base—not the old Anchor.
4. **Optional A-Only Hinge:** contrast continuation/citation decisions, without negative continuation Hinge at C1/C2 stop sites.

Main inference is standard greedy generation with **unmerged LoRA**, thinking disabled and `max_new_tokens=256`. Entity blocking is a separate baseline. Lexical support is a proxy, not a semantic evidence-exhaustion certificate. “EQUAL” is a project name; no formal acronym is asserted.

## Latest matched-baseline results

Full311, fixed training Seed `20261015`. Q, F1, repetition and true cap are percentages; tokens include generated special tokens. These are internal ALCE/QAMPARI-derived splits, not official leaderboard results.

| Qwen3-14B system | Q | Full F1 | Tokens | Repetition | True cap |
|---|---:|---:|---:|---:|---:|
| Base | 51.86 | 12.40 | 26.23 | 4.18 | 0.00 |
| Entity Blocker | 44.63 | 11.41 | 24.48 | 0.96 | 0.00 |
| Historical Anchor | 51.45 | 11.47 | 119.70 | 65.27 | 33.12 |
| Exhaustive-Prompt Base | **70.45** | 12.68 | 40.67 | 5.47 | 0.00 |
| Matched whole-answer SFT | 65.91 | 13.76 | 35.49 | 8.04 | 0.32 |
| Matched sequence DPO + 0.1 SFT | 65.70 | 13.75 | 38.21 | 8.36 | 0.32 |
| EQUAL CE, β=0 | 57.44 | 13.34 | 29.54 | 5.14 | 0.00 |
| EQUAL AOnly, β=0.25 | 60.33 | **13.78** | 31.17 | 6.43 | 0.32 |
| EQUAL AOnly, β=0.50 | 60.74 | 13.60 | 30.53 | 6.43 | 0.00 |

EQUAL β=0.25 minus exhaustive prompt / SFT / DPO has ΔQ **−10.12 / −5.58 / −5.37 percentage points**, with nominal 95% paired intervals **[−19.69,−1.12] / [−12.79,−0.70] / [−12.74,−0.39]**. Corresponding Full F1 difference intervals all include zero. A shorter output is a tradeoff, not proof of overall superiority or statistical equivalence.

SFT and EQUAL β=0.25 both lose 10 supported groups; SFT adds 78 vs EQUAL's 51. Higher teacher-forced prefix retention is **not** a guarantee of better test recall or fewer lost answer groups.

## Cross-family transfer and tokenizer audit

Mistral-7B-Instruct-v0.3, same Seed and 60-step protocol, using its own native training trajectories:

| System | Full311 Q | Full F1 | Tokens | Repetition | True cap |
|---|---:|---:|---:|---:|---:|
| Base Mistral | 37.81 | 8.99 | 46.91 | 2.25 | 0.64 |
| Matched SFT | **52.48** | **12.03** | 49.16 | 8.04 | 0.96 |
| EQUAL CE | 40.29 | 9.87 | 43.42 | 2.57 | 0.32 |
| EQUAL AOnly β=0.25 | 45.25 | 10.89 | 47.22 | 3.22 | 0.00 |

AOnly vs Base: ΔQ +7.44 pp **[+2.28,+13.49]**, ΔF1 +1.89 pp **[+1.03,+2.83]**. This supports transfer relative to Base, **not superiority over SFT**. AOnly−SFT Q CI crosses zero; its F1 CI is below zero. Mistral compilation yielded A=15, embedded C1=15, C2=10, and 120 retention trajectories / 5,317 positions.

Seven named tokenizers/encodings each showed boundary mismatch in **21/21** original A records. Qwen2.5/3 are not separate families; a duplicate local Mistral copy is not another family. Segmentation mismatch does **not** prove a universal hard decoding deadlock. See [Table S1 and literal/context IDs](results/supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md).

## Main three-Seed experiment: separate evidence

The earlier internally preregistered Full311 evaluation comprised 12 frozen systems / 3,732 generations, with zero new training or checkpoint selection. These are **three-Seed means ± sample SD**, not the single-Seed numbers above:

| Configuration | Full311 Q | Full F1 | Tokens |
|---|---:|---:|---:|
| CE β=0 | 56.40 ± 0.90 | 13.20 ± 0.17 | 29.24 ± 0.27 |
| AOnly β=0.25 | 58.13 ± 2.01 | 13.48 ± 0.26 | 30.05 ± 0.99 |
| AOnly β=0.50 | 57.99 ± 2.39 | 13.37 ± 0.22 | 29.89 ± 0.56 |

[Main report](results/reports_20261015/EQUAL_HELDOUT311_FINAL_RESULTS_20261015.md) · [Three Seeds and 12 ablations](results/reports_20261015/EQUAL_MULTI_SEED_ABLATION_RESULTS_20261015.md) · [Preregistration](results/reports_20261015/EQUAL_HELDOUT311_PREREGISTRATION_20261015.md).

The main evaluation was one-shot under an internal protocol, not third-party registration. Supplementary comparisons ran **after unsealing**; they are not a second blind test. None of these sets certify semantic near-duplicate or pretraining-contamination absence.

## CPU quick start

```bash
python scripts/show_results.py
python scripts/verify_release.py
python -m pip install -r requirements-analysis.txt
python scripts/run_checks.py
python scripts/reproduce_supplementary.py
```

The last command reconstructs **325 point values, 65 ledger summaries and 1,425 difference/CI values**, using the exact frozen statistical function bodies and 20,000 question-paired draws. It needs no GPU, raw passages, generated text, account or model download. Anonymous numeric sufficient statistics preserve the sampling order; this verifies statistics, **not correctness of text extraction or model generation**.

## Scope and reproducibility

- [Results index](results/README.md): all five views, main Seed details/hierarchical intervals, ablations, S1/S2/S3, gain/loss, diagnostics and checksums.
- [Current source](research/trajectory_repair_20261017/): dependency-closed dated implementations with configurable paths. Historical sources remain separate.
- [Reproduction guide](docs/REPRODUCIBILITY.md): exact settings and owner-supplied prepared-input requirements. GPU replay is **not turnkey** from distributed numeric results.
- No raw questions/passages, full generated answers, real-answer token traces, weights, logits, SSH helpers or credentials are distributed. See [data policy](docs/DATA.md).
- Updates/positive visits are matched; EQUAL additionally uses 240 auxiliary retention visits. This is **not equal FLOPs or controlled wall time**. No serving acceleration or semantic-entailment gain is established.
- Intervals are nominal/unadjusted and tied to their stated Seed/question target. CI crossing zero is not equivalence. New baselines/Mistral have one training Seed.

The [2026-10-11 ledger](docs/EXPERIMENTS_20261011.md) is historical: its isolated-endpoint loss, 0/72 native-stop finding and then-zero held-out predictions are not current implementation claims.

## License and citation

The owner has **not granted an open-source license**. Visibility does not grant general reuse permission. Third-party assets retain their terms. [LICENSE_STATUS.md](LICENSE_STATUS.md) · [SECURITY.md](SECURITY.md) · [CITATION.cff](CITATION.cff).

No accepted paper or DOI is claimed. Resources include [ALCE](https://github.com/princeton-nlp/ALCE), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), PyTorch, Transformers and PEFT. DPO follows [Rafailov et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html).

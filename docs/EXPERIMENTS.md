# Experiment ledger: current and historical evidence

Updated through 2026-10-17. Historical experiments remain archived; new controls change interpretation, not original measured values.

## Full reports and numerical artifacts

| Study | Scope | Report | Numerical data |
|---|---|---|---|
| Main frozen evaluation | 12 systems / 3,732 generations / 3 Seeds; Full311, Multi85, Sparse226 | [Main report](../results/reports_20261015/EQUAL_HELDOUT311_FINAL_RESULTS_20261015.md) | [Seed details, paired/hierarchical CIs](../results/heldout311_20261015/summary.json) |
| Development stability + ablation | Three configurations × three Seeds; 12 component ablations | [Report](../results/reports_20261015/EQUAL_MULTI_SEED_ABLATION_RESULTS_20261015.md) | [Summary](../results/multiseed_ablation_20261015/summary.json) |
| Supplement S1 | Seven named tokenizer/encoding audits on 21 A repairs | [S1/S2/S3 report](../results/supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md) | [Audit](../results/supplementary_20261017/tokenizer_boundary_audit.json) |
| Supplement S2 | Qwen strong prompt, matched SFT/DPO and frozen controls; five views | [Full report](../results/supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md) | [Metrics/CIs/ledgers](../results/supplementary_20261017/package2/summary.json) · [Anonymous cases](../results/supplementary_20261017/package2/question_statistics.json) |
| Supplement S3 | Mistral Base/SFT/CE/AOnly, native labels, five views | [Full report](../results/supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md) | [Metrics/CIs/ledgers](../results/supplementary_20261017/package3/summary.json) · [Anonymous cases](../results/supplementary_20261017/package3/question_statistics.json) |
| Historical 2026-10-11 | Reused n80 development views, isolated-site objective, 0/72 native stopping | [Historical ledger](EXPERIMENTS_20261011.md) | [Results index](../results/README.md) |

Main evaluation was internally preregistered. Supplements ran after unsealing: **not a second blind test**. New baselines/Mistral have one training Seed. All are internal ALCE/QAMPARI-derived partitions, not official leaderboard scores.

## Fixed views

| View | Questions | Supported groups |
|---|---:|---:|
| multi45 | 45 | 176 |
| raw24 | 24 | 8 |
| Full311 | 311 | 484 |
| Multi85 (`multi` in JSON) | 85 | 428 |
| Sparse226 (`sparse` in JSON) | 226 | 56 |

Full311 has141 positive-denominator and170 zero-support questions. Multi85 uses ≥2 known lexical support groups; Sparse226 uses0–1. Historical Raw24 is fixed, **not the identical threshold construction**: it contains one ≥2-group question. Zero-support questions remain in F1, length, repetition and cap. No output-based/post-result filtering is used.

## Findings constraining paper claims

- Qwen Full311 Seed20261015: AOnly0.25 Q60.33 vs prompt70.45, SFT65.91, DPO65.70. All three AOnly-minus-baseline Q intervals are below zero; F1 intervals cross zero. No superiority/equivalence claim follows.
- AOnly0.25 gains51/loses10/net41 supported groups; SFT78/10/68, DPO81/14/67, prompt107/17/90. Better training-prefix Ret./120 is not sufficient for lower test loss or higher coverage.
- Mistral AOnly Q37.81→45.25, ΔQ[+2.28,+13.49] pp and ΔF1[+1.03,+2.83] pp vs Base. SFT Q52.48/F1 12.03 remains stronger on point estimates. AOnly−SFT Q CI crosses zero; its F1 CI is below zero.
- All seven named tokenizers/encodings have21/21 boundary mismatches. This is segmentation evidence, not universal hard deadlock. Qwen2.5/3 are related; duplicate Mistral is not another family; unavailable Llama3/Gemma are not counted as passes.
- Compactness is an observed tradeoff, not serving speed or total training-cost evidence. Native Mistral labels are15A/15 embeddedC1/10C2; none were fabricated to match Qwen's41-record pool.

See the [Chinese interpretive addendum](../results/supplementary_20261017/SUPPLEMENTARY_INTERPRETATION_20261017.md) for all important positive/negative boundaries.

## Metrics and uncertainty

Q is micro supported, locally cited alias-group recall—not precision, entailment or complete evidence exhaustion. Full F1 is macro question-level answer F1 against all gold aliases, including answers absent from retrieval. Tokens are generated IDs including special tokens; repetition uses the frozen unit rule; true cap is256 reached without actual EOS.

Supplementary intervals reuse exact frozen statistical functions:20,000 paired question draws, question RNG20261010,95% percentiles. Undefined zero-denominator draws are excluded **only from Q**. Valid draw counts are retained. Main three-Seed/hierarchical intervals are separate targets. All comparisons are nominal/unadjusted; CI crossing zero is not equivalence.

## Verification and publication transformation

Final server consistency tests10/10 passed; Windows/Linux independent reproduction matched325 point values,65 ledger summaries,1,425 difference/CI values. Original local800/server54 protected files were unchanged; four frozen Qwen replays matched380/380 token paths. An initial packaging directory-mtime warning was recovered using a fixed file list, without rerunning models.

Published here: full aggregate values, anonymous per-question numeric statistics and ledger counts. Not published: raw text/answer token traces/weights. Private paths are redacted; original/exported SHA distinguish publication transformations from reruns. See [validation](VALIDATION.md) and [reproduction](REPRODUCIBILITY.md).

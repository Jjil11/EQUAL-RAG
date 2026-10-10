# Experiment ledger

Snapshot: 2026-10-11. Raw archives were locally verified; this repository distributes aggregate evidence, not source examples, predictions or weights. See [`results/README.md`](../results/README.md) for machine-readable files.

## Latest Qwen round: completed, not confirmed superior

Qwen3-14B, non-thinking, BF16, greedy decoding, maximum 256 new tokens. 350 training rows, 150 active repairs (75 stop / 75 complete), 72 native stop contexts, one epoch, seed20260930. Same-label/action-budget CE and UL controls. See [`configs/qwen_native_consistent.json`](../configs/qwen_native_consistent.json).

Both 80-question collections are reused development sets, disjoint by exact IDs and normalized question strings, not certified paraphrase-disjoint. They are different questions, not a paired retrieval-condition ablation. Original ALCE evaluation material was internally partitioned for this pilot; these scores are not official leaderboard results.

### Full matched-control table

Q and F1 are percentages. Repetition and true-cap rates are also percentages.

| Condition | System | Lexical Q | Mean tokens | Full-answer F1 | Repetition | True cap |
|---|---|---:|---:|---:|---:|---:|
| Original dev | Anchor | 47.285 | 42.513 | 33.664 | 6.25 | 2.50 |
| Original dev | Pair | 47.964 | 39.913 | 33.802 | 2.50 | 1.25 |
| Original dev | CE | 47.964 | 43.038 | 33.454 | 6.25 | 2.50 |
| Original dev | UL | 47.738 | 39.325 | 33.418 | 1.25 | 1.25 |
| Raw top-5 dev | Anchor | 40.580 | 122.538 | 6.083 | 73.75 | 36.25 |
| Raw top-5 dev | Pair | 47.826 | 131.375 | 6.397 | 67.50 | 41.25 |
| Raw top-5 dev | CE | 49.275 | 131.450 | 6.535 | 68.75 | 41.25 |
| Raw top-5 dev | UL | 47.826 | 132.875 | 6.399 | 68.75 | 41.25 |

### Pair minus anchor: 20,000 question-paired bootstrap draws

| Difference | Original dev: estimate [95% CI] | Raw top-5 dev: estimate [95% CI] |
|---|---|---|
| Lexical Q, percentage points | +0.679 [−1.263,+2.450] | +7.246 [0,+15.714] |
| Mean tokens | −2.600 [−9.000,+1.013] | +8.838 [−1.288,+20.150] |
| Relative utility | +8.04% [−0.66%,+25.46%] | +9.93% [−10.30%,+30.02%] |
| Full F1, percentage points | +0.138 [−0.662,+0.902] | +0.315 [−0.088,+0.982] |
| True cap, percentage points | −1.25 [−3.75,0] | +5.00 [0,+11.25] |

Raw-retrieval pair utility vs. CE: −2.89% [−13.80%,+6.20%]; vs. UL: +1.14% [−6.05%,+9.23%]. Original-dev pair vs. CE: +7.83% [−0.54%,+25.03%]; vs. UL: −1.01% [−3.55%,+1.44%]. No reliable pair-specific advantage has been established.

All three arms failed the utility-positive-in-both-conditions gate. Lexical-Q noninferiority vs. anchor is not semantic/answer-F1 noninferiority. A CI crossing zero does not prove equivalence. A post-hoc 100k-draw/48-comparison Bonferroni-percentile sensitivity analysis also failed to establish positive utility; it is neither an exact calibrated test nor a remedy for reused development data.

### Metric definitions

- **Lexical Q**: total locally cited, alias-matched answer–document edges divided by total annotated edges. Exact comma units prevent neighboring-citation credit but can mishandle internal commas. It is not entailment or universal grounded correctness.
- **Full-answer F1**: per-question answer precision/recall against all official alias groups, averaged over questions. Not restricted to answers covered by the selected evidence. Top-5-recall F1 is separately named in JSON.
- **Tokens**: actual generated IDs, including terminal special tokens, not word counts or retokenized visible text.
- **Utility**: Q / mean generated tokens; not latency or a monetary saving estimate.
- **True cap**: safety maximum reached without a terminating EOS. Historical token-threshold cap proxies are separate fields.
- **Zero-support strata**: Q/U are undefined without annotated lexical edges. Raw-dev has 56/80 such questions; their quality/token metrics are retained, not erased or forced to stop. In those 56, current pair mean tokens146.98, F1 0.128%, true cap48.21%.

## Training-site mechanism diagnostics

Four models ×150 examples ×two sites yielded1200 forwards, including288 repeated identical model-site computations, not1200 independent examples. Current pair EOS argmax61/150 at terminal sites comes predominantly from completed counterfactuals. Native stop EOS argmax is **0/72** for anchor/pair/CE/UL.

| System | Native mean EOS probability | Native EOS argmax | Terminal EOS argmax /150 |
|---|---:|---:|---:|
| Anchor | 0.000001125 | 0/72 | 2 |
| Pair, interpolated0.5 | 0.000009714 | 0/72 | 61 |
| CE, interpolated0.5 | 0.000008680 | 0/72 | 65 |
| UL, interpolated0.5 | 0.000008562 | 0/72 | 65 |

Un-interpolated pair: EOS argmax0/72, mean probability0.000043696. This training-only check rules out interpolation as the sole explanation, not all causal explanations. It did not select another checkpoint on development.

## Fixed automatic citation screen

DeBERTa checkpoint revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`; 1120 existing outputs, 943 unique premise/claim pairs, truncated0. Six basic and eight constructed relation controls passed. No human annotations, real-data calibration, official T5-XXL AutoAIS equivalence, or independent confirmation is claimed.

| Condition | Anchor support (%) | Pair support (%) | Pair−anchor, pp [95% CI] |
|---|---:|---:|---|
| Original dev | 65.933 | 65.171 | −0.762 [−2.364,+0.319] |
| Raw top-5 dev | 41.173 | 41.548 | +0.375 [−1.636,+3.273] |

These are mean per-question supported-output proportions, not gold-answer coverage. Parsing, relation reasoning, combined-document inputs and classifier error still matter. Seven-system precision/support tables are in JSON, without raw textual inputs. The unadapted Qwen's low format-sensitive score is not a fair stand-in for a strong trained baseline.

## Latency pilot

Same GPU, BF16/batch1, greedy/cap256, ABBA order, 16 fixed reused-dev questions (8/condition), 64 timed generations plus4 warmups; all output IDs match references. Timer covers CUDA-synchronized prefill/decode, excludes tokenization/retrieval/loading. Other GPUs were active; not exclusive whole-server reservation.

| Condition, n=8 | Anchor seconds | Pair seconds | Relative delay [95% CI] |
|---|---:|---:|---|
| Original dev subset | 2.542 | 2.744 | +7.96% [+1.52%,+15.50%] |
| Raw top-5 subset | 3.183 | 5.408 | +69.94% [−3.60%,+209.19%] |

No acceleration established. The small subset is not representative evidence that all raw-dev questions are70% slower. No production throughput, reliable p95, energy saving or training amortization claim.

## Historical and invalid versions

The [history ledger](HISTORY.md) distinguishes earlier pilots from current evidence. D/E/F aggregate summaries remain available; latest source/result provenance is in `release_manifest.json`. The native-terminal G round was invalidated for72 conflicting positive/negative paths and is **excluded from valid method scores**. Its code is retained only as an audit reference.

Older seeds, wrong-EOS results and prior ablations cannot confirm the changed current implementation. None of the exploratory results alone establish the promised independent method contribution. Latest-version sealed311 predictions0, actual human annotations0; the [roadmap](ROADMAP.md) lists remaining evidence.

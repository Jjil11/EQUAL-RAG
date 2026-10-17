# EQUAL-RAG Supplementary Results — 2026-10-17

Fixed seed 20261015. All newly trained adapters are unmerged, use 60 updates and 240 supervised record visits, and are evaluated regardless of outcome. The 311 questions were already unsealed in the main study; these are supplementary post-unsealing comparisons, not a second independent blind test.

## Protocol and interpretive limits

- Same frozen lexical scorer, normalization, gain/loss ledger, and Q-only paired question bootstrap; 20,000 resamples, seed 20261010, nominal unadjusted 95% intervals.
- Whole-answer CE covers every response token. DPO uses summed sequence log probabilities, β_DPO=0.1, frozen Base reference, and 0.1 whole-answer CE stabilization. The 20 C2 pairs are identical; they receive stabilization CE but no artificial preference labels.
- Optimizer updates, supervised positive targets, LoRA initialization and 240-record visit order are matched. EQUAL additionally uses 240 auxiliary retention visits; equal updates are not a claim of equal FLOPs or wall-clock cost.
- Retention for adapted models uses the full original teacher-forced contexts. For Exhaustive Base, an additional changed-prompt counterfactual is reported separately; it is not a same-prompt parameter-retention statistic.
- Tokenization mismatch does not prove universal inability to continue, a semantic evidence-exhaustion detector, or guaranteed benefit on every backbone.

## Table S1 — tokenizer boundary audit

| Tokenizer | Literal `]` IDs | Literal `],` IDs | Contextual native `]` ID | Contextual continuation first changed ID | Mismatch |
|---|---|---|---:|---:|---:|
| microsoft/Phi-3-mini-4k-instruct | [4514] | [21251] | 29962 | 1402 | 21/21 (100.00%) |
| mistralai/Mistral-7B-Instruct-v0.3 | [5477] | [12820] | 29561 | 1949 | 21/21 (100.00%) |
| openai-community/gpt2 | [60] | [4357] | 60 | 4357 | 21/21 (100.00%) |
| Qwen/Qwen2.5-7B-Instruct | [60] | [1125] | 60 | 1125 | 21/21 (100.00%) |
| Qwen/Qwen3-14B | [60] | [1125] | 60 | 1125 | 21/21 (100.00%) |
| cl100k_base | [60] | [1145] | 60 | 1145 | 21/21 (100.00%) |
| o200k_base | [60] | [2155] | 60 | 2155 | 21/21 (100.00%) |

Context IDs refer to `Alice [1], Bob [1]` versus its continuation. Standalone IDs can differ from in-context IDs (notably Mistral). Qwen2.5 and Qwen3 are not independent tokenizer families. A second local Mistral copy is a consistency check, not another family. Unavailable/gated Llama3 and Gemma are not counted as zero mismatches. Exact windows and provenance are archived.

## Table S2 — Qwen3-14B matched-budget comparisons

### multi45: 45 questions / 176 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-14B | 52.27 | 24.73 | 45.42 | 0.00 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Entity Blocker | 53.98 | 25.42 | 44.33 | 0.00 | 0.00 | 10 | 7 | +3 | 9 | 1 |
| Historical Anchor | 52.27 | 20.02 | 148.93 | 66.67 | 46.67 | 26 | 26 | +0 | 24 | 2 |
| EQUAL β=0 | 64.20 | 27.05 | 50.09 | 0.00 | 0.00 | 24 | 3 | +21 | 14 | 10 |
| EQUAL β=0.25 | 64.20 | 27.08 | 51.13 | 2.22 | 0.00 | 25 | 4 | +21 | 15 | 10 |
| EQUAL β=0.50 | 59.66 | 27.56 | 52.00 | 0.00 | 0.00 | 17 | 4 | +13 | 17 | 0 |
| Exhaustive-Prompt Base | 68.18 | 26.31 | 59.11 | 13.33 | 0.00 | 34 | 6 | +28 | 23 | 11 |
| Matched Whole-Answer SFT | 71.59 | 28.98 | 57.22 | 0.00 | 0.00 | 36 | 2 | +34 | 25 | 11 |
| Matched Sequence DPO + 0.1 SFT | 73.30 | 29.10 | 60.42 | 4.44 | 0.00 | 39 | 2 | +37 | 28 | 11 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Entity Blocker − Base Qwen3-14B | 1.70 [-4.28, 8.64] | 0.69 [-0.79, 2.77] | 20000 |
| Historical Anchor − Base Qwen3-14B | 0.00 [-11.76, 12.18] | -4.70 [-11.24, 1.29] | 20000 |
| EQUAL β=0 − Base Qwen3-14B | 11.93 [0.00, 26.42] | 2.33 [0.03, 5.21] | 20000 |
| EQUAL β=0.25 − Base Qwen3-14B | 11.93 [-0.55, 26.44] | 2.35 [0.03, 5.24] | 20000 |
| EQUAL β=0.50 − Base Qwen3-14B | 7.39 [0.51, 16.13] | 2.83 [0.41, 5.82] | 20000 |
| Exhaustive-Prompt Base − Base Qwen3-14B | 15.91 [4.05, 29.67] | 1.59 [-1.71, 4.86] | 20000 |
| Matched Whole-Answer SFT − Base Qwen3-14B | 19.32 [7.10, 33.17] | 4.25 [1.62, 7.48] | 20000 |
| Matched Sequence DPO + 0.1 SFT − Base Qwen3-14B | 21.02 [8.77, 34.74] | 4.37 [1.80, 7.38] | 20000 |
| EQUAL β=0 − Exhaustive-Prompt Base | -3.98 [-11.45, 3.21] | 0.74 [-1.46, 3.44] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | -7.39 [-12.90, -2.82] | -1.93 [-3.66, -0.58] | 20000 |
| EQUAL β=0 − Matched Sequence DPO + 0.1 SFT | -9.09 [-15.20, -3.98] | -2.05 [-3.71, -0.63] | 20000 |
| EQUAL β=0.25 − Exhaustive-Prompt Base | -3.98 [-10.73, 2.50] | 0.76 [-1.27, 3.40] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -7.39 [-12.79, -2.86] | -1.90 [-3.75, -0.48] | 20000 |
| EQUAL β=0.25 − Matched Sequence DPO + 0.1 SFT | -9.09 [-14.62, -4.42] | -2.02 [-3.67, -0.62] | 20000 |

### raw24: 24 questions / 8 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-14B | 87.50 | 5.64 | 21.38 | 8.33 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Entity Blocker | 87.50 | 5.55 | 23.54 | 0.00 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Historical Anchor | 75.00 | 5.22 | 135.71 | 66.67 | 41.67 | 0 | 1 | -1 | 0 | 0 |
| EQUAL β=0 | 87.50 | 5.64 | 25.71 | 4.17 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| EQUAL β=0.25 | 87.50 | 5.55 | 27.88 | 4.17 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| EQUAL β=0.50 | 87.50 | 5.64 | 28.46 | 8.33 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Exhaustive-Prompt Base | 87.50 | 3.75 | 40.04 | 8.33 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Matched Whole-Answer SFT | 87.50 | 5.64 | 32.42 | 8.33 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Matched Sequence DPO + 0.1 SFT | 75.00 | 4.62 | 35.88 | 4.17 | 0.00 | 0 | 1 | -1 | 0 | 0 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Entity Blocker − Base Qwen3-14B | 0.00 [0.00, 0.00] | -0.09 [-0.28, 0.00] | 19736 |
| Historical Anchor − Base Qwen3-14B | -12.50 [-20.00, 0.00] | -0.42 [-1.11, 0.00] | 19736 |
| EQUAL β=0 − Base Qwen3-14B | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19736 |
| EQUAL β=0.25 − Base Qwen3-14B | 0.00 [0.00, 0.00] | -0.09 [-0.28, 0.00] | 19736 |
| EQUAL β=0.50 − Base Qwen3-14B | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19736 |
| Exhaustive-Prompt Base − Base Qwen3-14B | 0.00 [0.00, 0.00] | -1.89 [-5.39, 0.00] | 19736 |
| Matched Whole-Answer SFT − Base Qwen3-14B | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19736 |
| Matched Sequence DPO + 0.1 SFT − Base Qwen3-14B | -12.50 [-100.00, 0.00] | -1.02 [-2.96, 0.00] | 19736 |
| EQUAL β=0 − Exhaustive-Prompt Base | 0.00 [0.00, 0.00] | 1.89 [0.00, 5.39] | 19736 |
| EQUAL β=0 − Matched Whole-Answer SFT | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 19736 |
| EQUAL β=0 − Matched Sequence DPO + 0.1 SFT | 12.50 [0.00, 100.00] | 1.02 [0.00, 2.96] | 19736 |
| EQUAL β=0.25 − Exhaustive-Prompt Base | 0.00 [0.00, 0.00] | 1.80 [0.00, 5.30] | 19736 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | 0.00 [0.00, 0.00] | -0.09 [-0.28, 0.00] | 19736 |
| EQUAL β=0.25 − Matched Sequence DPO + 0.1 SFT | 12.50 [0.00, 100.00] | 0.93 [0.00, 2.69] | 19736 |

### full311: 311 questions / 484 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-14B | 51.86 | 12.40 | 26.23 | 4.18 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Entity Blocker | 44.63 | 11.41 | 24.48 | 0.96 | 0.00 | 1 | 36 | -35 | 1 | 0 |
| Historical Anchor | 51.45 | 11.47 | 119.70 | 65.27 | 33.12 | 84 | 86 | -2 | 68 | 16 |
| EQUAL β=0 | 57.44 | 13.34 | 29.54 | 5.14 | 0.00 | 37 | 10 | +27 | 36 | 1 |
| EQUAL β=0.25 | 60.33 | 13.78 | 31.17 | 6.43 | 0.32 | 51 | 10 | +41 | 51 | 0 |
| EQUAL β=0.50 | 60.74 | 13.60 | 30.53 | 6.43 | 0.00 | 54 | 11 | +43 | 54 | 0 |
| Exhaustive-Prompt Base | 70.45 | 12.68 | 40.67 | 5.47 | 0.00 | 107 | 17 | +90 | 84 | 23 |
| Matched Whole-Answer SFT | 65.91 | 13.76 | 35.49 | 8.04 | 0.32 | 78 | 10 | +68 | 72 | 6 |
| Matched Sequence DPO + 0.1 SFT | 65.70 | 13.75 | 38.21 | 8.36 | 0.32 | 81 | 14 | +67 | 75 | 6 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Entity Blocker − Base Qwen3-14B | -7.23 [-13.01, -2.48] | -0.99 [-1.91, -0.27] | 20000 |
| Historical Anchor − Base Qwen3-14B | -0.41 [-9.20, 8.95] | -0.94 [-2.76, 0.83] | 20000 |
| EQUAL β=0 − Base Qwen3-14B | 5.58 [1.06, 10.37] | 0.94 [0.07, 1.83] | 20000 |
| EQUAL β=0.25 − Base Qwen3-14B | 8.47 [3.00, 14.32] | 1.38 [0.39, 2.39] | 20000 |
| EQUAL β=0.50 − Base Qwen3-14B | 8.88 [3.18, 14.93] | 1.20 [0.25, 2.17] | 20000 |
| Exhaustive-Prompt Base − Base Qwen3-14B | 18.60 [9.43, 28.11] | 0.28 [-1.02, 1.56] | 20000 |
| Matched Whole-Answer SFT − Base Qwen3-14B | 14.05 [6.39, 22.55] | 1.36 [0.28, 2.50] | 20000 |
| Matched Sequence DPO + 0.1 SFT − Base Qwen3-14B | 13.84 [5.87, 22.58] | 1.35 [0.24, 2.52] | 20000 |
| EQUAL β=0 − Exhaustive-Prompt Base | -13.02 [-22.54, -3.67] | 0.66 [-0.45, 1.81] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | -8.47 [-16.19, -2.30] | -0.41 [-1.24, 0.30] | 20000 |
| EQUAL β=0 − Matched Sequence DPO + 0.1 SFT | -8.26 [-16.04, -1.94] | -0.41 [-1.29, 0.38] | 20000 |
| EQUAL β=0.25 − Exhaustive-Prompt Base | -10.12 [-19.69, -1.12] | 1.10 [-0.02, 2.24] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -5.58 [-12.79, -0.70] | 0.02 [-0.76, 0.73] | 20000 |
| EQUAL β=0.25 − Matched Sequence DPO + 0.1 SFT | -5.37 [-12.74, -0.39] | 0.03 [-0.79, 0.78] | 20000 |

### multi: 85 questions / 428 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-14B | 53.27 | 38.52 | 38.69 | 0.00 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Entity Blocker | 45.33 | 34.95 | 34.66 | 0.00 | 0.00 | 1 | 35 | -34 | 1 | 0 |
| Historical Anchor | 50.93 | 35.01 | 132.33 | 60.00 | 36.47 | 69 | 79 | -10 | 56 | 13 |
| EQUAL β=0 | 58.64 | 40.89 | 45.28 | 2.35 | 0.00 | 32 | 9 | +23 | 32 | 0 |
| EQUAL β=0.25 | 62.38 | 42.28 | 47.11 | 3.53 | 0.00 | 47 | 8 | +39 | 47 | 0 |
| EQUAL β=0.50 | 62.85 | 42.17 | 47.56 | 4.71 | 0.00 | 50 | 9 | +41 | 50 | 0 |
| Exhaustive-Prompt Base | 73.60 | 40.93 | 59.00 | 7.06 | 0.00 | 101 | 14 | +87 | 81 | 20 |
| Matched Whole-Answer SFT | 68.22 | 42.85 | 54.67 | 4.71 | 1.18 | 72 | 8 | +64 | 68 | 4 |
| Matched Sequence DPO + 0.1 SFT | 68.69 | 43.51 | 54.01 | 4.71 | 0.00 | 76 | 10 | +66 | 72 | 4 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Entity Blocker − Base Qwen3-14B | -7.94 [-14.29, -2.58] | -3.58 [-6.64, -0.99] | 20000 |
| Historical Anchor − Base Qwen3-14B | -2.34 [-11.76, 7.63] | -3.51 [-9.40, 2.20] | 20000 |
| EQUAL β=0 − Base Qwen3-14B | 5.37 [0.24, 10.70] | 2.36 [-0.55, 5.17] | 20000 |
| EQUAL β=0.25 − Base Qwen3-14B | 9.11 [2.97, 15.64] | 3.76 [0.59, 6.92] | 20000 |
| EQUAL β=0.50 − Base Qwen3-14B | 9.58 [3.23, 16.36] | 3.65 [0.50, 6.77] | 20000 |
| Exhaustive-Prompt Base − Base Qwen3-14B | 20.33 [10.07, 30.72] | 2.41 [-1.93, 6.71] | 20000 |
| Matched Whole-Answer SFT − Base Qwen3-14B | 14.95 [6.44, 24.19] | 4.33 [0.71, 8.21] | 20000 |
| Matched Sequence DPO + 0.1 SFT − Base Qwen3-14B | 15.42 [6.55, 24.89] | 4.99 [1.28, 8.89] | 20000 |
| EQUAL β=0 − Exhaustive-Prompt Base | -14.95 [-25.37, -4.68] | -0.04 [-3.69, 3.52] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | -9.58 [-17.85, -2.72] | -1.96 [-4.85, 0.35] | 20000 |
| EQUAL β=0 − Matched Sequence DPO + 0.1 SFT | -10.05 [-18.33, -3.14] | -2.62 [-5.56, -0.15] | 20000 |
| EQUAL β=0.25 − Exhaustive-Prompt Base | -11.21 [-21.62, -1.41] | 1.35 [-2.16, 4.77] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -5.84 [-13.70, -0.49] | -0.57 [-3.19, 1.52] | 20000 |
| EQUAL β=0.25 − Matched Sequence DPO + 0.1 SFT | -6.31 [-14.19, -0.95] | -1.23 [-3.87, 0.87] | 20000 |

### sparse: 226 questions / 56 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-14B | 41.07 | 2.58 | 21.55 | 5.75 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Entity Blocker | 39.29 | 2.56 | 20.65 | 1.33 | 0.00 | 0 | 1 | -1 | 0 | 0 |
| Historical Anchor | 55.36 | 2.61 | 114.95 | 67.26 | 31.86 | 15 | 7 | +8 | 12 | 3 |
| EQUAL β=0 | 48.21 | 2.98 | 23.62 | 6.19 | 0.00 | 5 | 1 | +4 | 4 | 1 |
| EQUAL β=0.25 | 44.64 | 3.06 | 25.18 | 7.52 | 0.44 | 4 | 2 | +2 | 4 | 0 |
| EQUAL β=0.50 | 44.64 | 2.86 | 24.13 | 7.08 | 0.00 | 4 | 2 | +2 | 4 | 0 |
| Exhaustive-Prompt Base | 46.43 | 2.05 | 33.77 | 4.87 | 0.00 | 6 | 3 | +3 | 3 | 3 |
| Matched Whole-Answer SFT | 48.21 | 2.82 | 28.28 | 9.29 | 0.00 | 6 | 2 | +4 | 4 | 2 |
| Matched Sequence DPO + 0.1 SFT | 42.86 | 2.56 | 32.27 | 9.73 | 0.44 | 5 | 4 | +1 | 3 | 2 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Entity Blocker − Base Qwen3-14B | -1.79 [-6.00, 0.00] | -0.02 [-0.21, 0.13] | 20000 |
| Historical Anchor − Base Qwen3-14B | 14.29 [-1.85, 30.23] | 0.03 [-1.19, 1.26] | 20000 |
| EQUAL β=0 − Base Qwen3-14B | 7.14 [0.00, 16.07] | 0.41 [-0.11, 1.01] | 20000 |
| EQUAL β=0.25 − Base Qwen3-14B | 3.57 [-5.00, 12.50] | 0.48 [-0.17, 1.23] | 20000 |
| EQUAL β=0.50 − Base Qwen3-14B | 3.57 [-5.00, 12.50] | 0.28 [-0.27, 0.90] | 20000 |
| Exhaustive-Prompt Base − Base Qwen3-14B | 5.36 [-5.08, 16.00] | -0.52 [-1.34, 0.18] | 20000 |
| Matched Whole-Answer SFT − Base Qwen3-14B | 7.14 [-2.08, 17.24] | 0.24 [-0.30, 0.84] | 20000 |
| Matched Sequence DPO + 0.1 SFT − Base Qwen3-14B | 1.79 [-8.93, 12.50] | -0.02 [-0.67, 0.57] | 20000 |
| EQUAL β=0 − Exhaustive-Prompt Base | 1.79 [-10.00, 13.56] | 0.93 [0.25, 1.74] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | 0.00 [-7.02, 7.27] | 0.17 [-0.24, 0.62] | 20000 |
| EQUAL β=0 − Matched Sequence DPO + 0.1 SFT | 5.36 [-3.57, 15.09] | 0.42 [-0.04, 1.00] | 20000 |
| EQUAL β=0.25 − Exhaustive-Prompt Base | -1.79 [-14.55, 11.11] | 1.00 [0.24, 1.90] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -3.57 [-10.77, 3.33] | 0.24 [-0.24, 0.84] | 20000 |
| EQUAL β=0.25 − Matched Sequence DPO + 0.1 SFT | 1.79 [-7.41, 11.11] | 0.50 [-0.03, 1.18] | 20000 |

## Table S3 — Mistral-7B cross-family transfer

### multi45: 45 questions / 176 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Mistral-7B | 32.95 | 16.61 | 43.80 | 2.22 | 2.22 | 0 | 0 | +0 | 0 | 0 |
| Matched Whole-Answer SFT | 52.27 | 23.17 | 52.96 | 11.11 | 0.00 | 42 | 8 | +34 | 34 | 8 |
| EQUAL β=0 | 34.66 | 16.80 | 46.22 | 0.00 | 2.22 | 5 | 2 | +3 | 3 | 2 |
| EQUAL β=0.25 | 40.34 | 17.84 | 50.00 | 6.67 | 2.22 | 16 | 3 | +13 | 11 | 5 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Matched Whole-Answer SFT − Base Mistral-7B | 19.32 [7.93, 32.16] | 6.56 [1.99, 11.62] | 20000 |
| EQUAL β=0 − Base Mistral-7B | 1.70 [-1.28, 5.20] | 0.19 [-1.38, 1.58] | 20000 |
| EQUAL β=0.25 − Base Mistral-7B | 7.39 [1.64, 14.21] | 1.23 [-1.44, 4.31] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | -17.61 [-30.90, -5.98] | -6.37 [-11.39, -1.89] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -11.93 [-23.08, -1.93] | -5.33 [-9.85, -1.26] | 20000 |

### raw24: 24 questions / 8 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Mistral-7B | 50.00 | 2.85 | 42.29 | 4.17 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Matched Whole-Answer SFT | 62.50 | 4.26 | 36.92 | 12.50 | 0.00 | 2 | 1 | +1 | 2 | 0 |
| EQUAL β=0 | 50.00 | 2.76 | 37.50 | 4.17 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| EQUAL β=0.25 | 50.00 | 2.76 | 42.75 | 4.17 | 0.00 | 0 | 0 | +0 | 0 | 0 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Matched Whole-Answer SFT − Base Mistral-7B | 12.50 [0.00, 20.00] | 1.41 [0.00, 4.23] | 19736 |
| EQUAL β=0 − Base Mistral-7B | 0.00 [0.00, 0.00] | -0.09 [-0.28, 0.00] | 19736 |
| EQUAL β=0.25 − Base Mistral-7B | 0.00 [0.00, 0.00] | -0.09 [-0.28, 0.00] | 19736 |
| EQUAL β=0 − Matched Whole-Answer SFT | -12.50 [-20.00, 0.00] | -1.50 [-4.42, 0.00] | 19736 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -12.50 [-20.00, 0.00] | -1.50 [-4.42, 0.00] | 19736 |

### full311: 311 questions / 484 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Mistral-7B | 37.81 | 8.99 | 46.91 | 2.25 | 0.64 | 0 | 0 | +0 | 0 | 0 |
| Matched Whole-Answer SFT | 52.48 | 12.03 | 49.16 | 8.04 | 0.96 | 90 | 19 | +71 | 68 | 22 |
| EQUAL β=0 | 40.29 | 9.87 | 43.42 | 2.57 | 0.32 | 26 | 14 | +12 | 22 | 4 |
| EQUAL β=0.25 | 45.25 | 10.89 | 47.22 | 3.22 | 0.00 | 47 | 11 | +36 | 41 | 6 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Matched Whole-Answer SFT − Base Mistral-7B | 14.67 [7.28, 22.71] | 3.04 [1.82, 4.34] | 20000 |
| EQUAL β=0 − Base Mistral-7B | 2.48 [-0.76, 5.85] | 0.87 [-0.03, 1.80] | 20000 |
| EQUAL β=0.25 − Base Mistral-7B | 7.44 [2.28, 13.49] | 1.89 [1.03, 2.83] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | -12.19 [-20.04, -5.13] | -2.16 [-3.31, -1.08] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -7.23 [-16.17, 1.41] | -1.14 [-2.26, -0.06] | 20000 |

### multi: 85 questions / 428 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Mistral-7B | 40.89 | 31.12 | 61.75 | 4.71 | 0.00 | 0 | 0 | +0 | 0 | 0 |
| Matched Whole-Answer SFT | 54.44 | 37.38 | 65.25 | 8.24 | 1.18 | 76 | 18 | +58 | 55 | 21 |
| EQUAL β=0 | 43.22 | 33.57 | 56.52 | 3.53 | 0.00 | 23 | 13 | +10 | 19 | 4 |
| EQUAL β=0.25 | 48.13 | 36.48 | 62.27 | 5.88 | 0.00 | 41 | 10 | +31 | 35 | 6 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Matched Whole-Answer SFT − Base Mistral-7B | 13.55 [5.38, 22.65] | 6.26 [2.50, 10.15] | 20000 |
| EQUAL β=0 − Base Mistral-7B | 2.34 [-1.26, 6.10] | 2.45 [-0.70, 5.52] | 20000 |
| EQUAL β=0.25 − Base Mistral-7B | 7.24 [1.48, 13.98] | 5.36 [2.67, 8.21] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | -11.21 [-20.15, -3.31] | -3.81 [-7.21, -0.57] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -6.31 [-16.43, 3.28] | -0.90 [-4.23, 2.49] | 20000 |

### sparse: 226 questions / 56 support groups

| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Mistral-7B | 14.29 | 0.67 | 41.32 | 1.33 | 0.88 | 0 | 0 | +0 | 0 | 0 |
| Matched Whole-Answer SFT | 37.50 | 2.49 | 43.11 | 7.96 | 0.88 | 14 | 1 | +13 | 13 | 1 |
| EQUAL β=0 | 17.86 | 0.95 | 38.49 | 2.21 | 0.44 | 3 | 1 | +2 | 3 | 0 |
| EQUAL β=0.25 | 23.21 | 1.26 | 41.55 | 2.21 | 0.00 | 6 | 1 | +5 | 6 | 0 |

| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |
|---|---|---|---:|
| Matched Whole-Answer SFT − Base Mistral-7B | 23.21 [11.32, 35.59] | 1.83 [0.93, 2.84] | 20000 |
| EQUAL β=0 − Base Mistral-7B | 3.57 [-3.28, 11.11] | 0.28 [-0.07, 0.88] | 20000 |
| EQUAL β=0.25 − Base Mistral-7B | 8.93 [0.00, 18.46] | 0.59 [0.08, 1.29] | 20000 |
| EQUAL β=0 − Matched Whole-Answer SFT | -19.64 [-30.51, -9.68] | -1.54 [-2.46, -0.71] | 20000 |
| EQUAL β=0.25 − Matched Whole-Answer SFT | -14.29 [-24.00, -5.77] | -1.23 [-2.10, -0.48] | 20000 |

## Training and retention diagnostics

### Qwen3-14B

| System | Updates | Supervised visits | Auxiliary retention visits | Full-context Ret./120 | Training seconds |
|---|---:|---:|---:|---:|---:|
| Matched Whole-Answer SFT | 60 | 240 | 0 | 53/120 | 215.5 |
| Matched Sequence DPO + 0.1 SFT | 60 | 240 | 0 | 30/120 | 674.6 |

### Mistral-7B

Native compilation counts: `{"eligible_A": 15, "eligible_C2": 10, "native_EOS_trajectories": 348, "A": 15, "C1_embedded": 15, "C2": 10, "B": 0, "total_records": 25, "retention_trajectories": 120, "retention_positions": 5317, "selection": "fixed seed-hash caps 21 A,20 C2,120 retention; no test-score gate", "known_lexical_inventory_not_exhaustive_semantic_evidence": true}`.

| System | A first action | First new entity exact tokens | C1 closure | C2 stop | Retained trajectories |
|---|---:|---:|---:|---:|---:|
| base | 0/15 | 0/15 | 5/15 | 10/10 | 120/120 |
| matched_sft | 15/15 | 14/15 | 15/15 | 10/10 | 28/120 |
| equal_ce | 13/15 | 12/15 | 15/15 | 10/10 | 81/120 |
| equal_a025 | 15/15 | 12/15 | 15/15 | 10/10 | 73/120 |

Second-backbone identity: `${EQUAL_RAG_MISTRAL_MODEL}`. All weight/config/tokenizer file SHA-256 digests are in `package3/model_identity.json`. The 350 training rollouts, eligible/rejected label audit, deterministic caps, compiled records, full-prefix references and visit schedules are archived.

## Verification & checksums

All frozen source snapshots are checked before and after; no original checkpoint, prediction file, or main blind-test report is overwritten. Raw prediction digests:

| New prediction file | Rows | SHA-256 |
|---|---:|---|
| `package2/exhaustive_prompt/predictions380.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `package3/base/eval/independent_question_shards/gpu0/predictions.jsonl` | 95 | `9dbee216fdc35fbef9b946b95a0725917a984b1934e92aeb6bcf885241c2deae` |
| `package3/base/eval/independent_question_shards/gpu1/predictions.jsonl` | 95 | `a63fc89142b7735fc3e35315581a4e289c2e9097e182d0152af3e025a0fd2094` |
| `package3/base/eval/independent_question_shards/gpu2/predictions.jsonl` | 95 | `71e97706553ba0df336e309e5214a451a7c7d83e74a4284c474c5cc806706a64` |
| `package3/base/eval/independent_question_shards/gpu3/predictions.jsonl` | 95 | `fd3b8a6de177f45bcf9d50c21cfb1e8c95a99bb7dcd335ac6383a345bafe8c63` |
| `package3/base/eval/predictions380.jsonl` | 380 | `cd64315421e8a5e2f2445100493d1b8bfd855d25b309cca0b61f645261864755` |
| `package3/equal_a025/eval/predictions380.jsonl` | 380 | `4a63bc380cc2dd2bf11076ea35c5b2678934c363a3398c40dbfb3707233e6d55` |
| `package3/equal_ce/eval/predictions380.jsonl` | 380 | `4e0eb66b5c72e5fd89647c9bba1634c40254bb23eb3067799a38a51e1a5be70b` |
| `package3/matched_sft/eval/predictions380.jsonl` | 380 | `110cee0484718bc51694eb30488a9aae74cf4e063845dc1bf9f6340a275c1dcf` |
| `package3/native_train350/independent_question_shards/gpu0/predictions.jsonl` | 88 | `9ddf269291c89ab9e8af74fb651b77e1de6491e93620265ead8df6f4ad51f8b2` |
| `package3/native_train350/independent_question_shards/gpu1/predictions.jsonl` | 88 | `543349fb6e3463a71944801cacfb5c262b354994b4ef3c52deabc3c18dcc4e25` |
| `package3/native_train350/independent_question_shards/gpu2/predictions.jsonl` | 88 | `14e539923a1f43a57146326f7892aaa2071f9a6f53cdc8f2741016aed00eb1aa` |
| `package3/native_train350/independent_question_shards/gpu3/predictions.jsonl` | 86 | `173dba9b4c45ac6ad6476d8762bc2c3236ddbab9bc50136f54ef6a05cc75b287` |
| `package3/native_train350/predictions.jsonl` | 350 | `13197548de4af11f778276705c7761f32cf316ee5fbbab31ab25f5bdfc2030d3` |
| `single_gpu_v3/package2/exhaustive_prompt/predictions380.jsonl` | 380 | `0c11845702a8625c45f803bcc7ba153c6fd09735e724d5fa1f3495697e33bb53` |
| `single_gpu_v3/package2/matched_dpo/predictions380.jsonl` | 380 | `ce71099a692eb77cd1f289814eda72d50e3f5a180626836d08a9916a71351f42` |
| `single_gpu_v3/package2/matched_sft/predictions380.jsonl` | 380 | `ada71129cce68e33b2e8911ef266f683786e02b9564ef195006653981d31f7ef` |
| `single_gpu_v3/replay/a025/predictions380.jsonl` | 380 | `de1f87ee5597962fd5302c1af6b12aaec667f1375ee15c8fb31b0c7f8eb159b2` |
| `single_gpu_v3/replay/a050/predictions380.jsonl` | 380 | `4ca497ff29ebf08daacf3b17af3532f4f243c66a6775c0099d2263fcca1d73df` |
| `single_gpu_v3/replay/base/predictions380.jsonl` | 380 | `213da1811974b247df674e19f9c3cc24ca02c94130af032c309670ca4c8720d8` |
| `single_gpu_v3/replay/ce/predictions380.jsonl` | 380 | `2774a33077f04cae51e875f407faaf5ded4ad2b86395436e3e4da875e9fc0107` |
| `technical_recovery_v1/package2/exhaustive_prompt/predictions380.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `technical_recovery_v2/package2/exhaustive_prompt/predictions380.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Verification PASS: 54 protected frozen files unchanged; all requested sample counts/denominators and gain/loss identities checked. See `analysis/verification.json` and raw ledgers for full detail.

All positive and negative outcomes are retained. This single-seed supplementary study does not add three-seed evidence for the new baselines or Mistral; no universal cross-model claim follows from one additional backbone.

DPO source: https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
Mistral model card: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3


## Execution-placement reproducibility audit

Cross-device attempts stalled or produced non-finite/OOM failures. All original attempts and logs are preserved. Final Qwen evaluation uses one full A100 per process, BF16/SDPA and unmerged adapters. Hyperparameters and final checkpoints are unchanged. The completed 60-step DPO adapter is reused; the non-finite SFT attempt completed zero optimizer updates and is not a competing checkpoint.

| Frozen system replay | Identical token paths / 380 | Different token paths |
|---|---:|---:|
| Base Qwen3-14B | 380/380 | 0 |
| EQUAL β=0 | 380/380 | 0 |
| EQUAL β=0.25 | 380/380 | 0 |
| EQUAL β=0.50 | 380/380 | 0 |

Table S2 uses the replayed Base and frozen EQUAL adapters so the new-baseline causal comparisons use the same execution placement. Historical Anchor and Entity Blocker reuse original archived predictions; these are not matched-budget causal controls. Original frozen predictions and the original 311-question report remain untouched. Non-identical replays are disclosed, not used to select checkpoints or hyperparameters.

Exhaustive-Prompt cross-prompt counterfactual retention: 28/120 full original continuations retain every reference top-1 action when teacher-forced under the modified prompt. This is not identical-prompt parameter retention.

| Frozen EQUAL replay | Full-context Ret./120 |
|---|---:|
| Base Qwen3-14B | 120/120 |
| EQUAL β=0 | 93/120 |
| EQUAL β=0.25 | 89/120 |
| EQUAL β=0.50 | 91/120 |
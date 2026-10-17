# Data, models and publication policy

Distributed: dated research code, all aggregate metrics/intervals, anonymous per-question numeric sufficient statistics and attribution counts, tokenizer literal/synthetic IDs, settings, counts and hashes. Numeric supplementary data reproduce points/Bootstrap without source text.

Not distributed: credentials, server addresses/private paths, SSH/auth helpers, raw questions/passages, full generated answers, gold alias lists, real-answer token arrays/windows, model/adapter weights, cached logits, raw logs/archives or manuscript submission packages. Public visibility does not override upstream terms. No project open-source license has been granted.

Exports redact machine paths and text-recoverable actual-answer token traces. Tokenizer literal/synthetic Alice/Bob examples,21 mismatch decisions and original/config digests remain. Original/exported SHA distinguish release transformation from experimental reruns. Hidden text extraction cannot be independently validated from hashes or numeric sufficient statistics alone.

## Splits

Internal ALCE/QAMPARI-derived partitions, not official leaderboard splits:350 training, reused development,311 formerly sealed questions. Exact-ID/normalized-question overlap checks passed; semantic near-duplicates, document overlap, pretraining memory and unknown historical sources are not certified.

Full311:141 positive-support and170 zero-support questions,484 groups. Multi85≥2 groups; Sparse2260–1 groups. Reused multi45/raw24 have45/176 and24/8; Raw24 is not threshold-identical to Sparse226 (one ≥2-group question). No output-based filtering.

Main study: internal preregistration,12 frozen systems/3732 generations, no new training or checkpoint selection. Later baseline/Mistral comparisons use already unsealed311 data: supplementary held-out evaluation, not another blind test. No human semantic annotation is claimed.

## Upstream and required assets

- [ALCE](https://github.com/princeton-nlp/ALCE): obtain data under upstream data/source terms.
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B): official Base; owner-compiled labels/adapter assets separate.
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3): exact revision/files in model identity JSON.
- [Historical DeBERTa screen](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli): exploratory automatic screen, not official AutoAIS or human evaluation.

No upstream resource is relicensed. Gold is used in training compilation/evaluation, not inference. Lexical alias support can be relation-insensitive or incomplete; known inventory exhaustion is not semantic evidence exhaustion.

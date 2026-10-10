# Data and model asset policy

## What is distributed

Research source, synthetic examples, aggregate metrics/intervals, counts and hashes. No retrieved passages, raw dataset rows, question-level model outputs, blinded human materials, teacher logits, model or adapter weights are bundled.

The release manifest states original and exported hashes. An aggregate result file may be filtered to remove example-level labels/IDs, while preserving numeric summaries; that is a distribution transformation, not a rerun. Result summaries alone are insufficient to recompute question-paired bootstrap without the original paired examples.

## Upstream resources

- [ALCE](https://github.com/princeton-nlp/ALCE): upstream citation benchmark resources and QAMPARI/ASQA retrieval data. Obtain its data using the upstream instructions; consult applicable data/source terms, not only the repository code license.
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B): acquire separately, preserving tokenizer and generation configuration. Consult the model card and its license.
- [DeBERTa NLI checkpoint](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli): optional exploratory judge; fixed revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Not the official AutoAIS model.

No upstream repository or model is vendored/relicensed here. Raw retrieval inputs must not be reconstructed using gold reranking when claiming realistic retrieval performance.

## Split boundaries

The pilot partitioned material from an official ALCE evaluation source into its own train/development/sealed groups. That is disclosed internal development, **not official leaderboard evaluation**. Latest350train, original80dev and raw80dev are disjoint in exactIDs and normalized strings; semantic near-duplicate/document overlap is not certified.

Both80dev collections have been repeatedly inspected. The latest-version311sealed collection has zero model predictions in this snapshot. Do not tune variants on it or equate old-version runs with a newly frozen final confirmation.

## Prepared labels required by current training

The native-consistent training rows include `id`, `prompt`, `short`, evidence `edges`, `has_trajectory`, `completion_pair`, `action_decision`, `terminal_decision`, original `generation_token_ids`, `failure_local_retention_mask`, and prompt/response cache SHA metadata. The compiler audits350rows/150active/72native sites and exact positive-path conflicts.

These include tokenizer/model-dependent states, not ordinary SFT prompt/completion pairs. Synthetic examples demonstrate scoring/repair concepts but are **not substitutes for these prepared training rows**. Historical replay compilers assert exact input hashes. Different prepared data require a new, declared experiment protocol, not removal of audit asserts to force a run.

## Access and reuse

Private archive access is a separate owner decision. Do not publish original examples or model adapters before reviewing upstream conditions and manuscript anonymity. The repository's lack of an open-source license does not alter upstream obligations.

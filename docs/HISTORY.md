# Development history / 演化历程

## Current follow-up through2026-10-17

Native-tail repair, AOnly variants, three-Seed/12-ablation development and the internally preregistered311 evaluation are complete. New supplementary tokenizer, strong-prompt, matched SFT/DPO and Mistral experiments are also complete. Their negative controls limit superiority claims. Start from [current method](METHOD.md) and [current evidence](EXPERIMENTS.md).

## Preserved2026-10-11 history (not current status)

This ledger explains why the source tree contains many dated experiments. It is not a claim that every component is part of one validated final method.

| Stage | Question investigated | Current interpretation |
|---|---|---|
| Earlier token saving / SOLAR-related work | Can short-output post-training save tokens across models? | Historical non-RAG pilots; `historical/token_saving/`. Their results do not confirm the later EQUAL-RAG version. |
| Evidence-equivalent concise targets | Can omitted answers, swapped citations and repeated items teach complete brevity? | Established training/data pipeline; oracle-selected evidence is not realistic top-5 retrieval proof. |
| Anchor/quality protection and KL | Can shorter answers preserve useful behavior? | Multiple experimental variants; soft KL does not guarantee complete factual answers. |
| On-policy trajectory repair | Can model-visited repetition boundaries become repair targets? | Repetition boundaries can still be incomplete; “repeat detected” cannot automatically mean “safe to stop.” |
| Verified EOS and strict scoring | Are termination and local citation credit measured correctly? | Wrong post-EOS newline targets/neighbor citation credit invalidated some interpretations. Earlier three-seed results do not confirm corrected final code. |
| D clean labels | Do corrected labels improve reliable generation? | Complete exploratory round, but utility/strong-baseline evidence not sufficient. |
| E raw retrieval alignment | Does realistic top-5 training help? | 800 development outputs complete; first repair sites contained no EOS. Larger repair coverage did not establish efficient generation. |
| F terminal bridge | Does explicit EOS after completed repair fix stopping? | 480 outputs complete; completed teacher-forced prefixes often differ from native generation states. |
| G native-terminal-only replacement | Can original prefixes receive terminal EOS? | **Invalid:**72 first/terminal positive–negative path conflicts. No scores count as valid evidence. |
| H jointly consistent native supervision | Does removing those conflicts solve the task? | 480 outputs complete; exact conflicts0, but native EOS argmax0/72 and utility gate failed. Latest documented version, not a confirmed winner. |
| Automatic semantics / mechanism / latency | Are proxy improvements also semantically supported and faster? | Screening/diagnostics complete; no stable semantic gain or serving acceleration established. |
| Next proposal | Learn a coherent repair sequence instead of isolated endpoints | **Not launched** in this snapshot. Keep coverage protection and ordinary inference. |

## Navigating dated files

- Start from `train_qwen_native_consistent_20261010.py` and the matching compiler if inspecting the latest implementation.
- It reuses `train_qwen_terminal_bridge_20261010.py`, `train_qwen_priority_20261010.py`, `train_completion_repair_20261010.py`, and the downloaded original `train_trajectory_equal.py` base.
- D/E/F are preserved as historical implementations and aggregate result snapshots, not independent test replications.
- `compile_qwen_native_terminal_20261010.py` / `train_qwen_native_terminal_20261010.py` implement the invalid G version. Do not train or cite them as recommended methods.
- Several `prepare_`, `finalize_`, `audit_` and `test_` programs expect historical local assets. Not every archived script is a turnkey portable CLI.

Original sources were preserved locally. Release copies replace machine-specific paths with configurable expressions and record original/exported SHA256 values. An exported file can therefore differ byte-for-byte without being a newly trained method. Do not silently mix the release source hash with the historical training manifest's hash.

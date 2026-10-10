# Research roadmap

As of2026-10-11, current matched-control round execution is complete; method superiority is not established. Do not automatically expand seeds, models or sealed evaluation simply because a run finished.

## Immediate hypothesis (not launched)

Coherent repair-span supervision at actual model divergence points: valid syntactic closure, missing supported answer/citation spans, then EOS. Preserve existing evidence coverage with selective KL. Keep uncertain/zero-lexical-support examples protected rather than forcing stop. Use ordinary inference, no router.

Start with train-side token/path consistency checks and a small free-generation check. Declare one new version and compare matched CE/UL with comparable total supervision/forward budgets. Freeze the comparison before viewing results; report failures as well as positive points. A longer loss alone is not sufficient novelty.

## Six remaining evidence groups

1. **Stable Qwen method advantage** against trained anchor, matched/strong CE/UL and credible adapted efficient post-training baselines. Full-answer quality and semantic citations must accompany token utility.
2. **Independent confirmation**: freeze the final method/data/parameters; add two training seeds and once-only sealed311evaluation. Question uncertainty and seed variability are different quantities.
3. **Independent real Web retrieval task**: original retrieval ranking, no gold-aware inference/reranking, clearly disclosed official/internal splits.
4. **Trustworthy semantic citation evaluation**: calibrate a fixed judge on real relational/multi-hop examples and triangulate; automatic NLI is preliminary. Human annotations are currently0 and unavailable, not silently replaced by model judgments.
5. **Final-version mechanism and robustness**: protection/completion/EOS/KL-scope ablations; cap128/256/512 and missing/noisy retrieval evidence. Earlier-version ablations do not confirm revised code.
6. **System economics**: sufficiently large same-hardware latency/throughput measurements, realistic batch sizes, memory and training/cache amortization. The small replay did not establish speedup.

Prepare evaluation protocols and data audits in parallel, but prioritize group1 before expensive multi-model scaling. No “perfect support,” acceptance, or deadline guarantee is asserted.

For WWW relevance, frame a genuine Web information-access/evidence-accountability problem, not only a general language-model objective run on Wikipedia-derived data. Publication visibility and manuscript anonymity are owner decisions; the current repository remains private.

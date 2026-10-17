# Result index: current and historical

## Current studies through2026-10-17

- [Full S1/S2/S3 report](supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md) · [Chinese interpretation](supplementary_20261017/SUPPLEMENTARY_INTERPRETATION_20261017.md).
- [Qwen all views/CIs/ledgers](supplementary_20261017/package2/summary.json) · [anonymous numeric cases](supplementary_20261017/package2/question_statistics.json).
- [Mistral all views/CIs/ledgers](supplementary_20261017/package3/summary.json) · [anonymous numeric cases](supplementary_20261017/package3/question_statistics.json).
- [Tokenizer audit](supplementary_20261017/tokenizer_boundary_audit.json) · [native counts](supplementary_20261017/package3/counts.json) · [training diagnostics](supplementary_20261017/package3/training_diagnostics.json) · [manifests](supplementary_20261017/training_manifests.json).
- [Original prediction hashes](supplementary_20261017/verification.json) · [original independent checks](supplementary_20261017/local_verification.json) · [replay consistency](supplementary_20261017/execution_replay_consistency.json). These hashes refer to private original files, not bundled raw answers.
- [Main three-Seed Full311 summary](heldout311_20261015/summary.json) · [report](reports_20261015/EQUAL_HELDOUT311_FINAL_RESULTS_20261015.md) · [preregistration](reports_20261015/EQUAL_HELDOUT311_PREREGISTRATION_20261015.md).
- [Three-Seed +12 ablation summary](multiseed_ablation_20261015/summary.json) · [report](reports_20261015/EQUAL_MULTI_SEED_ABLATION_RESULTS_20261015.md).

Anonymous S2/S3 statistics reproduce paired20k CIs, not raw text scoring. `prediction_sha256` in summaries hashes canonical objects; raw JSONL hashes are in `verification.json`. Current published-file digests are in repository-root `publication_20261017.json`.

## Historical2026-10-11 aggregate files

| Directory | Contents |
|---|---|
| `qwen_native_consistent/` | Latest full numeric development table/comparisons and100k multiplicity sensitivity |
| `citation_nli/` | Fixed-revision seven-system support/precision screen, paired differences, warnings |
| `diagnostics/` | Aggregate native/teacher-forced EOS findings and un-interpolated diagnostic |
| `latency/` | Sixteen-question single-GPU ABBA pilot summary |
| `history/` | D clean-label / E raw-retrieval / F terminal-bridge aggregate summaries |

No raw output text, retrieved paragraphs, training rows or model weights are distributed. The latest primary `training_audit` is reduced to counts/hashes; historical top-level label audits may be omitted. Original/exported SHA values and transformations are documented in `../release_manifest.json`.

These are development results, not official leaderboard scores. Match condition, version, metric definition and comparator before reporting a number. A positive estimate with a CI crossing0 is not stable superiority. Latest-version sealed predictions0, actual human annotations0.

# Validation and limits

The completed original supplement passed10 server CPU tests: DPO identity/gradient, response-sum scoring, guard behavior, split counts, train/eval separation and native Mistral compiler fixture. Windows/Linux independent replay matched325 points,65 ledger summaries and1425 delta/CI values;23 raw prediction-file hashes/row counts passed. Five new models' configurations, within-backbone initialization/pool/order and60×4 visits were checked. Original local800/server54 protected files were unchanged; four frozen Qwen replays matched380/380 token paths.

The first tar returned a directory-mtime warning; fixed-file-list recovery succeeded without changing predictions or retraining. The historical packaging failure is not an unfinished model experiment.

## Public checks

The publication copy passed **25/25 CPU tests** and the complete numeric replay: **325 point values, 65 ledger summaries and 1,425 difference/CI values**, with 20,000 paired draws. These checks used Python 3.11 and NumPy 2.2.6; they did not run new training or inference.

- `scripts/verify_release.py`: old/new publication digests, Python syntax, common secret/private-path patterns, forbidden binaries/raw fields and local Markdown links. Pattern scanning is not an absolute secret guarantee.
- `scripts/run_checks.py`: historical synthetic metric/compiler tests plus current publication/invariant tests; no private data or GPU launches.
- `scripts/reproduce_supplementary.py`: reconstruct325 points/65 ledgers/1425 deltas/CI values from anonymous numeric cases using exact archived functions. This verifies statistics, not original text extraction, generation or semantic support.

The original19-test publication count belongs to2026-10-11. Main three-Seed/hierarchical and development ablation aggregates are provided, but the S2/S3 replay command does not claim to rerun those earlier GPU experiments. Local checks do not imply remote CI success; read GitHub workflow status separately.

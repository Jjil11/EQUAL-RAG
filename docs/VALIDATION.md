# Publication validation

Local validation completed on2026-10-11 using Windows/Python3.11.9 and NumPy2.4.6 in an isolated virtual environment. This is different from the original Linux/CUDA training environment.

- `scripts/show_results.py`: successfully displays the saved eight-row matched-control table, paired utility intervals, false expansion gate, and zero latest-version sealed/human counts.
- `scripts/verify_release.py`: PASS for153 source/aggregate manifest entries and149 parsed Python files; local Markdown links and common privacy/artifact checks passed.
- `scripts/run_checks.py`: **19 tests passed**: six strict-metric tests, nine synthetic completion/compiler invariants, four publication scope/integrity checks.
- Result JSON was checked for unexpected raw-example fields; no question/prediction/premise/claim payload is included in the distributed aggregates.

This does not validate a full14B end-to-end fresh installation, GPU numerics, every archived data-dependent test, real-data NLI calibration or claimed method superiority. NumPy dependency ranges are not an exact historical analysis lock. No remote CI status is represented by these local results.

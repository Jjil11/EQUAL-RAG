# Contributing to the research snapshot

The repository is private and has no open-source license yet. Coordinate access and reuse with the owner before contributing or redistributing.

For an experiment change, state the hypothesis, exact training/data version, frozen comparator, resource budget, random seeds, and evaluation protocol before running. Use a new run directory rather than overwriting completed runs. Separate development results from untouched confirmation; do not tune on the sealed set.

Run `python scripts/verify_release.py` and, with NumPy installed, `python scripts/run_checks.py`. Add a source/result provenance entry for new snapshots. Keep invalid experiments identifiable, report all matched controls, and distinguish lexical support from semantic entailment.

Never commit raw third-party examples, weights, credential files, or unreviewed logs. Discuss changes using an issue or branch; a documentation edit must not silently change historical numeric evidence.

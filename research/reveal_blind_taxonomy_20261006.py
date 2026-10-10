#!/usr/bin/env python3
"""Reveal a frozen blind audit and summarize method-side behavior."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "artifacts/blinded_taxonomy_20261006"
ann = {r["blind_id"]: r for r in map(json.loads, (DIR / "blind_annotations.jsonl").read_text(encoding="utf-8").splitlines())}
keys = {r["blind_id"]: r for r in map(json.loads, (DIR / "identity_key.jsonl").read_text(encoding="utf-8").splitlines())}
assert ann.keys() == keys.keys()

def verdict(pref, method_side):
    if pref in ("tie", "none", "not_judged_without_evidence"):
        return pref
    if pref == "mixed":
        return "mixed"
    return "win" if pref == method_side else "loss"

def loss_side(label):
    if label == "yes":
        return "B"
    if label.startswith("yes_if_"):
        return label[-1]
    if label == "changed_set":
        return "changed_set"
    return "none"

stats = defaultdict(lambda: defaultdict(int))
revealed = []
for bid, row in ann.items():
    key = keys[bid]
    method = "A" if key["A"] == "method" else "B"
    a = row["annotation"]
    concise = verdict(a["preferred_for_conciseness"], method)
    coverage = verdict(a["visible_coverage_proxy"], method)
    rep = verdict(a["repetition_removed"], method)
    aloss = loss_side(a["answer_loss"])
    method_loss = aloss == method
    stratum = row["sampling_stratum"]
    stats[stratum]["n"] += 1
    stats[stratum][f"conciseness_{concise}"] += 1
    stats[stratum][f"coverage_{coverage}"] += 1
    stats[stratum][f"repetition_{rep}"] += 1
    stats[stratum]["method_visible_answer_loss"] += int(method_loss)
    stats[stratum]["changed_answer_set"] += int(aloss == "changed_set")
    revealed.append({
        "blind_id": bid, "stratum": stratum, "family": key["family"], "seed": key["seed"],
        "method_side": method, "quality_delta": key["quality_delta"], "token_delta": key["token_delta"],
        "method_conciseness": concise, "method_visible_coverage": coverage,
        "method_repetition_removal": rep, "method_visible_answer_loss": method_loss,
        "answer_set_changed": aloss == "changed_set",
    })

summary = {
    "protocol": "annotations frozen before identity key download; output-only structural audit",
    "factual_grounding_judged": False,
    "n": len(revealed),
    "by_stratum": {k: dict(v) for k, v in sorted(stats.items())},
}
(DIR / "revealed_annotations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in revealed), encoding="utf-8")
(DIR / "reveal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))

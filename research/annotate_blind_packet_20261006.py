#!/usr/bin/env python3
"""Apply a frozen, identity-blind structural audit to the 2026-10-06 packet.

The packet intentionally contains no retrieved evidence, so the audit does not
pretend to judge factual grounding.  It records visible coverage, repetition,
brevity, and citation-bearing answer loss; method identities are revealed only
after this file has been produced.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/blinded_taxonomy_20261006/blind_packet.jsonl"
DST = ROOT / "artifacts/blinded_taxonomy_20261006/blind_annotations.jsonl"

# preferred_for_grounded_completeness is deliberately "not_judged": retrieved
# passages were withheld from the packet, so a factual preference would be an
# unsupported claim.  coverage_proxy records only visible unique-answer scope.
RATINGS = {
    "1c8428502372": ("A", "B", "B", "yes", "yes", "A"),
    "15b09798ab99": ("tie", "B", "B", "no", "no", "A"),
    "7cfd309bb64f": ("tie", "B", "B", "no", "no", "A"),
    "b55103d31edd": ("A", "B", "B", "yes", "yes", "A"),
    "eedfe5a0d019": ("B", "B", "B", "no", "no", "A"),
    "6186ef4d3aad": ("A", "A", "A", "yes_if_B", "yes_if_B", "B"),
    "f47db29af2d4": ("A", "B", "B", "yes", "yes", "A"),
    "6dbe2127dd62": ("mixed", "B", "B", "changed_set", "changed_set", "A"),
    "b9bbbe0cfc7c": ("tie", "A", "A", "no", "no", "B"),
    "ce93902aec0f": ("A", "B", "B", "yes", "yes", "A"),
    "35a1a7aa088a": ("mixed", "B", "B", "changed_set", "changed_set", "A"),
    "561184cf3f72": ("tie", "tie", "none", "no", "no", "none"),
    "cc84a2febaff": ("tie", "tie", "none", "no", "no", "none"),
    "3531fda63b12": ("tie", "tie", "none", "no", "no", "none"),
    "98e0c7a9844c": ("tie", "tie", "none", "no", "no", "none"),
    "72c4c87bb07d": ("tie", "tie", "none", "no", "no", "none"),
    "868ded697af8": ("tie", "tie", "none", "no", "no", "none"),
    "d7a05d45ee71": ("tie", "tie", "none", "no", "no", "none"),
    "cc1feafad874": ("tie", "tie", "none", "no", "no", "none"),
    "59b44bcc6fa9": ("tie", "tie", "none", "no", "no", "none"),
    "b22a89f19198": ("tie", "tie", "none", "no", "no", "none"),
    "300e1872223b": ("tie", "tie", "none", "no", "no", "none"),
    "5e415e289175": ("tie", "tie", "none", "no", "no", "none"),
    "53d707be8829": ("B", "A", "none", "yes_if_A", "yes_if_A", "none"),
    "98c1daed98d5": ("B", "A", "none", "yes_if_A", "yes_if_A", "none"),
    "a4a615cc9b72": ("A", "B", "none", "yes_if_B", "yes_if_B", "none"),
    "17bad72c6a8c": ("B", "A", "none", "yes_if_A", "yes_if_A", "none"),
    "6bb33ec16b7d": ("A", "B", "none", "yes_if_B", "yes_if_B", "none"),
    "2a1d080fc1e1": ("A", "B", "none", "yes_if_B", "yes_if_B", "none"),
    "0af5a879d932": ("A", "B", "none", "changed_set", "changed_set", "none"),
    "0d76aba7a098": ("B", "A", "none", "yes_if_A", "yes_if_A", "none"),
    "7dbaa53fd97b": ("A", "B", "none", "changed_set", "changed_set", "none"),
    "3a7e1fb7a58e": ("B", "A", "none", "yes_if_A", "yes_if_A", "none"),
    "b220d1aa5a2a": ("A", "B", "none", "yes_if_B", "yes_if_B", "none"),
    "e7747164490f": ("A", "B", "none", "yes_if_B", "yes_if_B", "none"),
}

rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines()]
assert {r["blind_id"] for r in rows} == set(RATINGS), "packet/rating mismatch"
for row in rows:
    coverage, concise, removed, answer_loss, citation_loss, semantic_dup = RATINGS[row["blind_id"]]
    row["annotation"] = {
        "preferred_for_grounded_completeness": "not_judged_without_evidence",
        "visible_coverage_proxy": coverage,
        "preferred_for_conciseness": concise,
        "repetition_removed": removed,
        "answer_loss": answer_loss,
        "citation_loss": citation_loss,
        "semantic_duplication": semantic_dup,
        "notes": "Identity-blind output-only structural audit; factual grounding requires retrieved passages.",
    }
DST.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(json.dumps({"output": str(DST), "n": len(rows), "identity_revealed": False}, ensure_ascii=False))

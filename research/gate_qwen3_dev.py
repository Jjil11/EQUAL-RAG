#!/usr/bin/env python3
"""Gate a Qwen3 A-EQUAL adapter before any test-set evaluation."""

import json
import sys
from pathlib import Path


short = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
aequal = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
quality_floor = short["grounded_edge_recall"] - 0.01
passed = (
    aequal["grounded_edge_recall"] >= quality_floor
    and aequal["mean_output_tokens"] < short["mean_output_tokens"]
    and aequal["grounded_utility_per_token"] > short["grounded_utility_per_token"]
)
report = {
    "quality_floor": quality_floor,
    "short_stop": short,
    "aequal_kl": aequal,
    "passes": passed,
}
print(json.dumps(report, indent=2))
raise SystemExit(0 if passed else 2)

"""Read-only format inspection of completed development outputs; no gold tuning."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
import re
from pathlib import Path
from completion_pairs_20261010 import units
ROOT=Path(str(_release_root))
RUN=ROOT/'runs/qwen_retrieval_align_20261010'

def main():
    for setting in ['regression_dev','prospective_dev']:
        for arm in ['foundation','action_pair','action_ce','action_ul','strong_ce']:
            path=RUN/arm/setting/'eval_results.jsonl'
            if not path.exists():continue
            rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            if len(rows)!=80:continue
            report={'setting':setting,'arm':arm,'n':80,
                'no_numeric_citations':sum(not re.search(r'\[\d+\]',r['output']) for r in rows),
                'bracket_comma_citations':sum(bool(re.search(r'\[\d+\s*,\s*\d+',r['output'])) for r in rows),
                'comma_inside_closed_unit':sum(any(',' in u for _,_,u in units(r['output'])) for r in rows)}
            if arm=='foundation':report['first_three_outputs']=[{'id':r['id'],'tokens':r['tokens'],'output':r['output'][:1000]} for r in rows[:3]]
            print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

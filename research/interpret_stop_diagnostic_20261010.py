"""Derive actual greedy stop margins from saved one/two-EOS telemetry; no model use."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
import math
from pathlib import Path

RUN=Path(str(_release_root / 'runs/submission_20261010'))
folder=RUN/'stop_probability_diagnostic'
report={'scope':'Original active training prefixes, not heldout examples',
        'warning':'Improved logits do not demonstrate a greedy stopping decision. No test predictions or weights are changed.',
        'families':{}}
for family in ['qwen3','mistral']:
    path=folder/f'{family}_prefixes.jsonl'
    assert (folder/f'{family}_status.txt').read_text().strip()=='COMPLETE'
    records=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    arms={x['arm'] for x in records}
    table={}
    for arm in sorted(arms):
        rows=[x for x in records if x['arm']==arm]
        margins=[]
        for row in rows:
            assert len(row['eos_ids']) in [1,2], 'Cannot reconstruct individual EOS masses for larger classes'
            best_non_logp=row['canonical_eot_logp']-row['canonical_margin']
            canonical=math.exp(row['canonical_eot_logp'])
            max_eos=canonical
            if len(row['eos_ids'])==2:
                max_eos=max(canonical,max(0.,row['total_stop_probability']-canonical))
            margins.append(math.log(max(1e-300,max_eos))-best_non_logp)
        table[arm]={'n':len(rows),'greedy_stop_winning_prefixes':sum(x>=0 for x in margins),
                    'maximum_greedy_stop_margin':max(margins),
                    'mean_greedy_stop_margin':sum(margins)/len(margins),
                    'canonical_target_winning_prefixes':sum(x['canonical_margin']>=0 for x in rows)}
    report['families'][family]=table
(folder/'greedy_reachability_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

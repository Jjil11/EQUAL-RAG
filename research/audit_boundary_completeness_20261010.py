"""Training-only safety audit: first repetition does not imply completeness."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
from pathlib import Path
from score_strict_units_20261010 import norm
ROOT=Path(str(_release_root))


def matched(text,edges):
    units={norm(x) for x in text.rstrip().rstrip('.').split(',') if norm(x)}
    return {i for i,e in enumerate(edges) if units.intersection(norm(a) for a in e['aliases'])}


def main():
    report={'warning':'Training diagnostics only. First repeated unit is a heuristic failure boundary, not a completeness certificate.'}
    for family,file in [('mistral','train_trajectory_onpolicy_seed20260930.jsonl'),
                        ('qwen3','train_trajectory_qwen3_14b_onpolicy_seed20260930.jsonl')]:
        rows=[json.loads(x) for x in (ROOT/'data/qampari_evidence_full'/file).read_text().splitlines()]
        active=[x for x in rows if x.get('has_trajectory')]
        stats=[]
        for x in active:
            before=matched(x['trajectory_prefix'],x['edges']); full=matched(x['short'],x['edges'])
            stats.append({'id':x['id'],'gold_edges':len(x['edges']),'prefix_matched_edges':len(before),
                          'full_matched_edges':len(full),'new_supported_answers_after_first_repeat':len(full-before),
                          'cap_hit':x.get('anchor_cap_hit',False)})
        report[family]={'active':len(active),'all_gold_covered_at_boundary':sum(x['prefix_matched_edges']==x['gold_edges'] for x in stats),
                        'new_supported_answers_later':sum(x['new_supported_answers_after_first_repeat']>0 for x in stats),
                        'later_supported_answer_total':sum(x['new_supported_answers_after_first_repeat'] for x in stats),
                        'active_cap_hit':sum(x['cap_hit'] for x in stats),'per_row':stats}
    out=ROOT/'runs/submission_20261010/boundary_completeness_audit.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_row'} if isinstance(v,dict) else v for k,v in report.items()},indent=2))


if __name__=='__main__':main()

"""Question-clustered metric audit of six legacy runs; no new model inference."""
import json
from pathlib import Path
import numpy as np
from build_blinded_taxonomy import ROOT,RUNS,read
from score_strict_units_20261010 import score,summarize


def main():
    data={x['id']:x for x in read(ROOT/'data/qampari_evidence_full/test.jsonl')}
    raw=json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text())
    official={str(x.get('id',x.get('sample_id',i))):x['answers'] for i,x in enumerate(raw)}
    for key,row in data.items():
        row['all_gold_answers']=official[key]
    report={'warning':'All Qwen legacy repair runs use post-EOT newline as boundary target; diagnostic results, not semantic-EOT repair evidence.',
            'answer_f1_denominator':'All official answer alias groups, not only oracle-selected evidence edges.',
            'runs':[],'families':{}}
    families={}
    for family,seed,bp,mp in RUNS:
        base={x['id']:x for x in read(ROOT/bp)};method={x['id']:x for x in read(ROOT/mp)}
        assert base.keys()==method.keys() and len(base)==159
        keys=sorted(base)
        b=[score(data[k],base[k]) for k in keys];m=[score(data[k],method[k]) for k in keys]
        record={'family':family,'seed':seed,'baseline':summarize(b),'method':summarize(m),
                'window_minus_strict_baseline':sum(x['legacy_grounded_edges']-x['grounded_edges'] for x in b),
                'window_minus_strict_method':sum(x['legacy_grounded_edges']-x['grounded_edges'] for x in m),
                'difference_warning':'Includes both citation-window scope and exact-unit/normalization differences; not solely citation borrowing.'}
        report['runs'].append(record)
        arrays=[np.array([[r['grounded_edges'],r['gold_edges'],r['tokens'],r['answer_f1']] for r in v],float) for v in [m,b]]
        families.setdefault(family,[]).append(arrays)
    for family,runs in families.items():
        a=np.stack([x[0] for x in runs]); b=np.stack([x[1] for x in runs])
        def calc(v):
            quality=v[:,:,0].sum(1)/v[:,:,1].sum(1)
            tokens=v[:,:,2].mean(1)
            return np.array([quality.mean(),tokens.mean(),(quality/tokens).mean(),v[:,:,3].mean()])
        observed=calc(a)-calc(b);rng=np.random.default_rng(20261010);boot=[]
        for _ in range(20000):
            ix=rng.integers(0,159,159);boot.append(calc(a[:,ix,:])-calc(b[:,ix,:]))
        boot=np.array(boot)
        report['families'][family]={'unique_questions':159,'seeds':3,'resampling':'question clusters; all three seeds retained together',
            'delta':dict(zip(['strict_grounded_quality','tokens','utility','official_style_f1'],observed.tolist())),
            'ci95':dict(zip(['strict_grounded_quality','tokens','utility','official_style_f1'],np.quantile(boot,[.025,.975],axis=0).T.tolist())),
            'quality_one_sided_lower95':float(np.quantile(boot[:,0],.05)),
            'quality_noninferior_at_margin_001':bool(np.quantile(boot[:,0],.05)>-.01),
            'not_independent_seed_inference':'Only three fixed seeds: intervals reflect question uncertainty, not new-seed generalization.'}
    out=ROOT/'runs/submission_20261010/legacy_metric_audit_all_answers.json'
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

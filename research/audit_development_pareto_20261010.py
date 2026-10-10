"""All completed frozen interpolation endpoints; no holdout or parameter tuning."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
from pathlib import Path
from score_strict_units_20261010 import score,summarize,paired
ROOT=Path(str(_release_root))
RUN=ROOT/'runs/submission_20261010'


def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    data={x['id']:x for x in read(ROOT/'data/qampari_evidence_full/dev.jsonl')}
    raw=json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text())
    answers={str(x.get('id',x.get('sample_id',i))):x['answers'] for i,x in enumerate(raw)}
    for k,row in data.items():row['all_gold_answers']=answers[k]
    report={'type':'Exploratory development Pareto audit; not a selected final-test comparison',
            'answer_metric':'ALCE-style list F1 against all official answer groups','families':{}}
    for family in ['qwen3','mistral']:
        variants={}
        for arm in ['semantic_eos','unit_ul']:
            for weight,sub in [('1.0','train'),('0.5','interp05/dev')]:
                p=RUN/f'{family}_{arm}'/sub/'eval_results.jsonl'
                if p.exists():variants[f'{arm}_w{weight}']=[score(data[x['id']],x) for x in read(p)]
        report['families'][family]={'points':{k:summarize(v) for k,v in variants.items()}}
        if len(variants)==4:
            comparisons={}
            for w in ['0.5','1.0']:
                comparisons[f'same_weight_{w}']=paired(variants[f'semantic_eos_w{w}'],variants[f'unit_ul_w{w}'])
            # Explicitly exploratory quality-matched candidate, alongside every
            # other point; never silently substitute it for the primary arm.
            comparisons['eos05_minus_ul10_exploratory']=paired(variants['semantic_eos_w0.5'],variants['unit_ul_w1.0'])
            report['families'][family]['comparisons']=comparisons
    (RUN/'development_pareto_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{'points':v['points'],'comparisons_available':'comparisons' in v} for k,v in report['families'].items()},indent=2),flush=True)


if __name__=='__main__':main()

"""Post-hoc conservative interval sensitivity, no new model inference.

The family is declared here after viewing development results; this is not
preregistration, and cannot remove development-selection bias.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from qwen_clean_labels_20261010 import ROOT,RUN
from score_strict_units_20261010 import score
from finalize_qwen_priority_20261010 import read

OUT=ROOT/'runs/qwen_clean_multiplicity_20261010'
NAMES=['grounded_quality','mean_tokens','relative_utility','answer_f1','repetition_rate','true_cap_hit_rate']
FAMILY=2*4*len(NAMES)


def main():
    if OUT.exists():raise FileExistsError('Refusing supplemental overwrite')
    primary=json.loads((RUN/'development_summary.json').read_text())
    assert primary['status']=='COMPLETE_CLEAN_TARGET_QWEN_DEVELOPMENT'
    official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in
              enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    report={'status':'COMPLETE_POSTHOC_MULTIPLICITY_SENSITIVITY','bootstrap_samples':100000,
            'seed':20261010,'family_size':FAMILY,'familywise_alpha':.05,
            'interval_tail_quantile':.05/(2*FAMILY),'model_generations':0,'heldout_predictions':0,
            'warnings':['Post-hoc family, not preregistration; does not remove reused-development selection bias.',
                        'Bonferroni bootstrap percentile interval sensitivity, not an exact calibrated significance test.',
                        'Single training seed; no inference about seed variance.'],
            'primary_summary_sha256':hashlib.sha256((RUN/'development_summary.json').read_bytes()).hexdigest(),
            'settings':{}}
    for setting in ['regression_dev','prospective_dev']:
        p=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
        data={r['id']:r for r in read(p)}
        if setting=='regression_dev':
            for k,r in data.items():r['all_gold_answers']=official[k]
        keys=sorted(data);arrays={}
        for arm in ['action_pair','baseline','action_ce','action_ul','strong_ce']:
            folder=RUN/arm/setting
            if arm=='baseline':folder=ROOT/'runs/submission_20261010/qwen3_frozen_comparison'/setting/arm
            if arm in ['action_ul','strong_ce']:folder=ROOT/'runs/qwen_clean_controls_20261010'/arm/setting
            preds={r['id']:r for r in read(folder/'eval_results.jsonl')}
            telemetry={r['id']:r for r in read(folder/'termination_telemetry.jsonl')}
            assert len(preds)==len(telemetry)==80 and set(preds)==set(data)==set(telemetry)
            values={k:score(data[k],preds[k]) for k in keys}
            arrays[arm]=np.array([[values[k]['grounded_edges'],values[k]['gold_edges'],values[k]['tokens'],
                                  values[k]['answer_f1'],values[k]['repeated_units']>0,
                                  telemetry[k]['hit_cap_without_eos']] for k in keys],float)
        def metrics(v):
            q=v[...,0].sum(-1)/np.maximum(1,v[...,1].sum(-1));t=v[...,2].mean(-1)
            return np.stack([q,t,q/np.maximum(1,t),v[...,3].mean(-1),v[...,4].mean(-1),v[...,5].mean(-1)],-1)
        comparisons={}
        for control in ['baseline','action_ce','action_ul','strong_ce']:
            rng=np.random.default_rng(20261010);boots=[];a=arrays['action_pair'];b=arrays[control]
            observed=metrics(a)-metrics(b);observed[2]=metrics(a)[2]/max(1e-12,metrics(b)[2])-1
            for _ in range(100):
                ix=rng.integers(0,len(keys),(1000,len(keys)))
                ma,mb=metrics(a[ix]),metrics(b[ix]);delta=ma-mb
                delta[:,2]=ma[:,2]/np.maximum(1e-12,mb[:,2])-1
                boots.append(delta)
            boot=np.concatenate(boots);tail=report['interval_tail_quantile']
            comparisons[control]={name:{'delta':float(observed[i]),
                'ordinary_ci95':np.quantile(boot[:,i],[.025,.975]).tolist(),
                'bonferroni_family_interval':np.quantile(boot[:,i],[tail,1-tail]).tolist()}
                for i,name in enumerate(NAMES)}
        report['settings'][setting]=comparisons
    OUT.mkdir()
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

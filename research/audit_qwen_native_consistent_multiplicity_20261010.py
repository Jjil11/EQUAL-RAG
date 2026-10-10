"""Conservative post-hoc interval sensitivity, no model inference."""
import hashlib,json
import numpy as np
from compile_qwen_native_consistent_20261010 import ROOT,RUN,SOURCE
from finalize_qwen_priority_20261010 import read
from score_strict_units_20261010 import score
OUT=ROOT/'runs/qwen_native_consistent_multiplicity_20261010'
NAMES=['grounded_quality','mean_tokens','relative_utility','answer_f1','repetition_rate','true_cap_hit_rate']

def main():
    if OUT.exists():raise FileExistsError('Refusing supplemental overwrite')
    path=RUN/'development_summary.json';primary=json.loads(path.read_text())
    assert primary['status']=='COMPLETE_QWEN_NATIVE_CONSISTENT_DEVELOPMENT'
    official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    report={'status':'COMPLETE_POSTHOC_MULTIPLICITY_SENSITIVITY','bootstrap_samples':100000,'seed':20261010,
        'family_size':48,'familywise_alpha':.05,'interval_tail_quantile':.05/96,'model_generations':0,'heldout_predictions':0,
        'primary_summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'warnings':['Post-hoc family and reused development: not preregistered confirmation.',
                    'Bonferroni percentile bootstrap sensitivity is not an exact calibrated test.',
                    'Intervals do not estimate training-seed variation.'],'settings':{}}
    for setting in ['regression_dev','prospective_dev']:
        p=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
        data={r['id']:r for r in read(p)}
        if setting=='regression_dev':
            for k,r in data.items():r['all_gold_answers']=official[k]
        keys=sorted(data);arrays={}
        for arm in ['action_pair','baseline','action_ce','action_ul','strong_ce']:
            folder=RUN/arm/setting
            if arm=='baseline':folder=ROOT/'runs/submission_20261010/qwen3_frozen_comparison'/setting/arm
            if arm=='strong_ce':folder=ROOT/'runs/qwen_retrieval_align_20261010/strong_ce'/setting
            preds={r['id']:r for r in read(folder/'eval_results.jsonl')};tel={r['id']:r for r in read(folder/'termination_telemetry.jsonl')}
            assert len(preds)==len(tel)==80 and set(preds)==set(tel)==set(data)
            v={k:score(data[k],preds[k]) for k in keys}
            arrays[arm]=np.array([[v[k]['grounded_edges'],v[k]['gold_edges'],v[k]['tokens'],v[k]['answer_f1'],v[k]['repeated_units']>0,tel[k]['hit_cap_without_eos']] for k in keys],float)
        def metrics(v):
            q=v[...,0].sum(-1)/np.maximum(1,v[...,1].sum(-1));t=v[...,2].mean(-1)
            return np.stack([q,t,q/np.maximum(1,t),v[...,3].mean(-1),v[...,4].mean(-1),v[...,5].mean(-1)],-1)
        comparisons={}
        for control in ['baseline','action_ce','action_ul','strong_ce']:
            rng=np.random.default_rng(20261010);a=arrays['action_pair'];b=arrays[control];observed=metrics(a)-metrics(b)
            observed[2]=metrics(a)[2]/max(1e-12,metrics(b)[2])-1;boots=[]
            for _ in range(100):
                ix=rng.integers(0,len(keys),(1000,len(keys)));ma,mb=metrics(a[ix]),metrics(b[ix]);delta=ma-mb
                delta[:,2]=ma[:,2]/np.maximum(1e-12,mb[:,2])-1;boots.append(delta)
            boot=np.concatenate(boots);tail=report['interval_tail_quantile']
            comparisons[control]={name:{'delta':float(observed[i]),'ordinary_ci95':np.quantile(boot[:,i],[.025,.975]).tolist(),
                'bonferroni_family_interval':np.quantile(boot[:,i],[tail,1-tail]).tolist()} for i,name in enumerate(NAMES)}
            for i,name in enumerate(NAMES):
                key='relative_delta' if name=='relative_utility' else 'delta'
                assert np.isclose(observed[i],primary['settings'][setting]['comparisons']['action_pair_minus_'+control][name][key])
        report['settings'][setting]=comparisons
    OUT.mkdir();(OUT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

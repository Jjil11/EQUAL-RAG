"""Complete development comparisons; vectorized question-paired bootstrap."""
import hashlib
import json
from pathlib import Path
import numpy as np
from completion_pairs_20261010 import ROOT, valid_training_edges, true_repeat
from score_strict_units_20261010 import score,summarize
from finalize_verified_all_answers_20261010 import relative_utility
from qwen_priority_compile_20261010 import RUN

ARMS=['seq_full','seq_local','action_pair','action_ce']


def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def paired_fast(method,base,telemetry_method,telemetry_base):
    m={r['id']:r for r in method};b={r['id']:r for r in base}
    assert set(m)==set(b) and len(m)==len(method) and len(b)==len(base)
    keys=sorted(m);ix=np.random.default_rng(20261010).integers(0,len(keys),(20000,len(keys)))
    fields=['grounded_edges','gold_edges','tokens','answer_f1','repeated_units']
    def array(d):return np.array([[d[k][f] for f in fields] for k in keys],float)
    a,z=array(m),array(b)
    def values(v):
        q=v[...,0].sum(-1)/np.maximum(1,v[...,1].sum(-1));t=v[...,2].mean(-1)
        return np.stack([q,t,q/np.maximum(1,t),v[...,3].mean(-1),(v[...,4]>0).mean(-1)],-1)
    observed=values(a)-values(z);boots=values(a[ix])-values(z[ix])
    names=['grounded_quality','mean_tokens','utility','answer_f1','repetition_rate']
    result={k:{'delta':float(observed[i]),'ci95':np.quantile(boots[:,i],[.025,.975]).tolist()}
            for i,k in enumerate(names)}
    lower=float(np.quantile(boots[:,0],.05))
    result['quality_noninferiority']={'margin':-.01,'one_sided_95_lower':lower,'supported':lower>-.01}
    result['relative_utility']=relative_utility(method,base)
    caps=np.array([int(telemetry_method[k]['hit_cap_without_eos'])-
                   int(telemetry_base[k]['hit_cap_without_eos']) for k in keys],float)
    result['true_cap_hit_rate']={'delta':float(caps.mean()),'ci95':np.quantile(caps[ix].mean(1),[.025,.975]).tolist()}
    return result


def main():
    target=RUN/'development_summary.json'
    if target.exists():raise FileExistsError('Frozen summary exists')
    train=read(RUN/'compiled/train.jsonl');train_ids={r['id'] for r in train}
    audit=json.loads((RUN/'compiled/audit.json').read_text())
    assert len(train)==len(train_ids)==350
    assert hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest()==audit['compiled_sha256']
    old=ROOT/'runs/submission_20261010';previous=ROOT/'runs/completion_repair_20261010'
    historical=json.loads((previous/'completion_inventory.json').read_text())
    for rel,expected in historical.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
    official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in
              enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    report={'status':'COMPLETE_QWEN_DEVELOPMENT_PILOT','model':'Qwen3-14B','seed':20260930,
            'holdout_predictions':0,'human_annotations':0,'training_audit':audit,
            'historical_files_verified':len(historical),'settings':{},'warnings':[
                'Reused development sets, no multiple-comparison correction; not final confirmation.',
                'Q is a lexical local citation proxy, not semantic entailment or full task correctness.',
                'The raw retrieval complement is not a separate independent benchmark.',
                'Mean suffix loss versus action loss have different scale and supervision.',
                'action_pair versus action_ce is forward-matched, not gradient-norm matched.',
                'Legal action candidates cover a bounded subset of acceptable expressions.']}
    assert (RUN/'diagnostic_status.txt').read_text().strip()=='COMPLETE'
    report['training_action_diagnostic']=json.loads((RUN/'action_diagnostic_summary.json').read_text())
    report['training_label_consistency']={'positive_completions_with_surface_repeat':sum(
        bool(true_repeat({'short':r['completion_pair']['chosen']})) for r in train if r.get('has_trajectory')),
        'warning':'Subsequent train-only audit found seven contradictory positive targets; retained as historical controls, not a final clean-label method.'}
    report['warnings'].append('Seven positive completions contain repeated names despite lexical edge coverage; clean-target follow-up is separate.')
    for arm in ARMS:
        folder=RUN/arm
        assert (folder/'status.txt').read_text().strip()=='COMPLETE'
        manifest=json.loads((folder/'train/train_manifest.json').read_text())
        assert manifest['seed']==20260930 and manifest['epochs']==1 and manifest['train_examples']==350
        assert manifest['lr']==1e-5 and manifest['trajectory_alpha']==.1 and manifest['anchor_alpha']==10
        a=json.loads((folder/'train/priority_training_audit.json').read_text())
        assert a['seen_rows']==350 and a['active_pairs']==39
    for setting in ['regression_dev','prospective_dev']:
        source=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
        data={r['id']:r for r in read(source)};assert len(data)==80 and not train_ids.intersection(data)
        if setting=='regression_dev':
            for k,row in data.items():row['all_gold_answers']=official[k]
        results={};telemetry={};table={}
        for arm in ['baseline','unit_ul','semantic_eos','previous_pair','previous_sft']+ARMS:
            if arm in ARMS:folder=RUN/arm/setting
            elif arm.startswith('previous_'):folder=previous/('qwen3_'+arm.split('_',1)[1])/setting
            else:folder=old/'qwen3_frozen_comparison'/setting/arm
            preds=read(folder/'eval_results.jsonl');tel=read(folder/'termination_telemetry.jsonl')
            assert len(preds)==len(tel)==80 and {r['id'] for r in preds}==set(data)
            assert len({r['id'] for r in preds})==len({r['id'] for r in tel})==80
            telemetry[arm]={r['id']:r for r in tel}
            assert all(r['tokens']==len(telemetry[arm][r['id']]['generated_token_ids']) for r in preds)
            results[arm]=[score(data[r['id']],r) for r in preds]
            table[arm]=summarize(results[arm]);table[arm]['true_cap_hit_rate']=float(np.mean([r['hit_cap_without_eos'] for r in tel]))
        comparisons={}
        pairs=[(a,b) for a in ARMS for b in ['baseline','unit_ul','previous_pair']]+[
            ('seq_local','seq_full'),('action_pair','action_ce'),('action_pair','seq_local')]
        for a,b in pairs:comparisons[a+'_minus_'+b]=paired_fast(results[a],results[b],telemetry[a],telemetry[b])
        strata={}
        for label,nonzero in [('nonzero_lexical_support',True),('zero_lexical_support',False)]:
            ids={k for k,r in data.items() if bool(valid_training_edges(r['edges']))==nonzero}
            if ids:strata[label]={'n':len(ids),'table':{a:summarize([r for r in rows if r['id'] in ids]) for a,rows in results.items()}}
        report['settings'][setting]={'n':80,'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'table':table,'comparisons':comparisons,'input_support_strata':strata}
    report['development_gates']={}
    for arm in ARMS:
        comparisons=[s['comparisons'][arm+'_minus_baseline'] for s in report['settings'].values()]
        gate={'quality_noninferior_both':all(c['quality_noninferiority']['supported'] for c in comparisons),
              'utility_ci_positive_both':all(c['relative_utility']['ci95'][0]>0 for c in comparisons),
              'no_clear_f1_loss_both':all(c['answer_f1']['ci95'][1]>=0 for c in comparisons),
              'no_clear_token_or_cap_regression_both':all(c['mean_tokens']['ci95'][0]<=0 and c['true_cap_hit_rate']['ci95'][0]<=0 for c in comparisons)}
        gate['ready_for_confirmatory_expansion']=all(gate.values())
        report['development_gates'][arm]=gate
    target.write_text(json.dumps(report,indent=2))
    inventory={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*')
               if p.is_file() and 'adapter' not in p.parts and p.name!='completion_inventory.json'}
    for pattern in ['*qwen_priority*20261010*']:
        for p in (ROOT/'code').glob(pattern):
            if p.is_file():inventory[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    reportfile=ROOT/'logs/qwen_priority_protocol_20261010.md'
    inventory[str(reportfile.relative_to(ROOT))]=hashlib.sha256(reportfile.read_bytes()).hexdigest()
    (RUN/'completion_inventory.json').write_text(json.dumps(inventory,indent=2))
    print(json.dumps({'tables':{k:s['table'] for k,s in report['settings'].items()},'gates':report['development_gates']},indent=2),flush=True)


if __name__=='__main__':main()

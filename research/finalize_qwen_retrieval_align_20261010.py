"""Audit complete raw-retrieval training pilot and paired development results."""
import hashlib
import json
from pathlib import Path
import numpy as np
from build_qwen_retrieval_align_20261010 import ROOT,RUN
from completion_pairs_20261010 import valid_training_edges
from finalize_qwen_priority_20261010 import read,paired_fast
from score_strict_units_20261010 import score,summarize

ARMS=['action_pair','action_ce','action_ul','strong_ce']


def main():
    target=RUN/'development_summary.json'
    if target.exists():raise FileExistsError('Refusing frozen summary overwrite')
    old=ROOT/'runs/qwen_clean_labels_20261010';controls=ROOT/'runs/qwen_clean_controls_20261010'
    old_inventory=json.loads((old/'completion_inventory.json').read_text())
    for rel,sha in old_inventory.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
    audit=json.loads((RUN/'compiled/audit.json').read_text())
    train=read(RUN/'compiled/train.jsonl');ids={r['id'] for r in train}
    assert len(train)==len(ids)==350 and audit['rows']==350 and audit['active']>=8
    assert hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest()==audit['compiled_sha256']
    assert audit['zero_lexical_support']==106 and audit['heldout_inputs_read']==0
    assert (RUN/'foundation/status.txt').read_text().strip()=='COMPLETE'
    for arm in ARMS:
        p=RUN/arm;assert (p/'status.txt').read_text().strip()=='COMPLETE'
        manifest=json.loads((p/'train/train_manifest.json').read_text())
        m=json.loads((p/'train/retrieval_alignment_manifest.json').read_text())
        a=json.loads((p/'train/clean_training_audit.json').read_text())
        assert manifest['train_examples']==a['seen_rows']==350 and a['active_pairs']==audit['active']
        assert a['action_counts']=={'stop':audit['stop'],'complete_then_stop':audit['completion']}
        assert manifest['epochs']==1 and manifest['seed']==20260930 and manifest['lr']==1e-5
        assert manifest['trajectory_alpha']==m['repair_alpha']==(.2 if arm=='strong_ce' else .1)
        assert m['arm']==arm and m['compiled_data_sha256']==audit['compiled_sha256']
    official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in
              enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    report={'status':'COMPLETE_QWEN_RETRIEVAL_ALIGNMENT_DEVELOPMENT','model':'Qwen3-14B','seed':20260930,
            'new_predictions':800,'holdout_predictions':0,'human_annotations':0,
            'previous_clean_round_files_verified_unchanged':len(old_inventory),'training_audit':audit,
            'data_manifest':json.loads((RUN/'data/manifest.json').read_text()),'settings':{},'warnings':[
                'Reused 80+80 development questions; single seed and no multiple-comparison correction.',
                'Lexical unit citation recall is not semantic entailment or full answer correctness.',
                'Rank-only train documents align retrieval mode, not query distribution; zero-edge fraction 30.3% train vs70% raw dev.',
                'Internal controlled partition of an official evaluation source, not official benchmark or external task.',
                'All training rows retained; unknown zero-edge repetitions preserve teacher KL and have no forced-stop supervision.',
                'Different fault counts confound old-vs-new comparisons of training condition and repair strength.',
                'Baseline is a trained anchor; finite candidates and same repair forwards do not prove grammar or gradient-strength equivalence.']}
    for setting in ['regression_dev','prospective_dev']:
        datafile=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
        data={r['id']:r for r in read(datafile)};assert len(data)==80 and not ids.intersection(data)
        if setting=='regression_dev':
            for k,r in data.items():r['all_gold_answers']=official[k]
        folders={a:RUN/a/setting for a in ARMS}
        folders['foundation']=RUN/'foundation'/setting
        folders.update({a:ROOT/'runs/submission_20261010/qwen3_frozen_comparison'/setting/a for a in ['baseline','unit_ul']})
        folders.update({'old_'+a:old/a/setting for a in ['action_pair','action_ce','seq_pair','seq_sft']})
        folders.update({'old_'+a:controls/a/setting for a in ['action_ul','strong_ce']})
        results={};telemetry={};table={}
        for arm,folder in folders.items():
            preds=read(folder/'eval_results.jsonl');tel=read(folder/'termination_telemetry.jsonl')
            assert len(preds)==len(tel)==80 and {r['id'] for r in preds}=={r['id'] for r in tel}==set(data)
            telemetry[arm]={r['id']:r for r in tel};assert len(telemetry[arm])==80
            assert all(r['tokens']==len(telemetry[arm][r['id']]['generated_token_ids']) for r in preds)
            results[arm]=[score(data[r['id']],r) for r in preds];table[arm]=summarize(results[arm])
            table[arm]['true_cap_hit_rate']=float(np.mean([r['hit_cap_without_eos'] for r in tel]))
        pairs=[(a,'baseline') for a in ARMS]+[(a,'old_'+a) for a in ARMS]
        pairs += [('action_pair',b) for b in ['action_ce','action_ul','strong_ce','unit_ul','old_seq_pair','old_seq_sft']]
        pairs += [(a,'foundation') for a in ['baseline']+ARMS]
        comparisons={a+'_minus_'+b:paired_fast(results[a],results[b],telemetry[a],telemetry[b]) for a,b in pairs}
        strata={}
        for label,nonzero in [('nonzero_lexical_support',True),('zero_lexical_support',False)]:
            selected={k for k,r in data.items() if bool(valid_training_edges(r['edges']))==nonzero}
            if selected:
                strata[label]={'n':len(selected),'table':{a:summarize([r for r in rows if r['id'] in selected]) for a,rows in results.items()},
                    'comparisons':{a+'_minus_baseline':paired_fast([r for r in results[a] if r['id'] in selected],
                        [r for r in results['baseline'] if r['id'] in selected],telemetry[a],telemetry['baseline']) for a in ARMS}}
                if not nonzero:
                    reason='No constructed lexical edges: Q/U and their noninferiority cannot evaluate this stratum.'
                    strata[label]['warning']=reason
                    for t in strata[label]['table'].values():
                        t['grounded_quality']=None;t['utility']=None
                    for c in strata[label]['comparisons'].values():
                        for k in ['grounded_quality','utility','relative_utility','quality_noninferiority']:
                            c[k]={'status':'UNDEFINED_FOR_ZERO_EDGE_STRATUM','reason':reason}
        report['settings'][setting]={'n':80,'data_sha256':hashlib.sha256(datafile.read_bytes()).hexdigest(),
                                    'table':table,'comparisons':comparisons,'input_support_strata':strata}
    report['development_gates']={}
    for arm in ARMS:
        values=[s['comparisons'][arm+'_minus_baseline'] for s in report['settings'].values()]
        gate={'quality_noninferior_both':all(v['quality_noninferiority']['supported'] for v in values),
              'utility_ci_positive_both':all(v['relative_utility']['ci95'][0]>0 for v in values),
              'no_clear_f1_loss_both':all(v['answer_f1']['ci95'][1]>=0 for v in values),
              'no_clear_token_or_cap_regression_both':all(v['mean_tokens']['ci95'][0]<=0 and v['true_cap_hit_rate']['ci95'][0]<=0 for v in values)}
        gate['ready_for_confirmatory_expansion']=all(gate.values());report['development_gates'][arm]=gate
    (RUN/'coordinator_status.txt').write_text('COMPLETE\n')
    target.write_text(json.dumps(report,indent=2))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*')
           if p.is_file() and 'adapter' not in p.parts and p.name!='completion_inventory.json'}
    for p in (ROOT/'code').glob('*qwen_retrieval*20261010*'):
        if p.is_file():files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'logs/qwen_retrieval_align_20261010_protocol.md';files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RUN/'completion_inventory.json').write_text(json.dumps(files,indent=2))
    members=set(files)|{str((RUN/'completion_inventory.json').relative_to(ROOT))}
    members.update(str(p.relative_to(ROOT)) for p in (ROOT/'logs/qwen_retrieval_align_20261010').rglob('*') if p.is_file())
    (ROOT/'exports/qwen_retrieval_align_20261010_members.list').write_text('\n'.join(sorted(members))+'\n')
    print(json.dumps({'tables':{k:s['table'] for k,s in report['settings'].items()},'gates':report['development_gates']},indent=2),flush=True)


if __name__=='__main__':main()

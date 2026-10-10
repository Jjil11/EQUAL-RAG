"""Completed explicit terminal supervision, paired development comparison."""
import hashlib,json,time
import numpy as np
from compile_qwen_terminal_bridge_20261010 import ROOT,RUN,SOURCE
from completion_pairs_20261010 import valid_training_edges
from finalize_qwen_priority_20261010 import read,paired_fast
from score_strict_units_20261010 import score,summarize
ARMS=['action_pair','action_ce','action_ul']

def main():
    target=RUN/'development_summary.json'
    if target.exists():raise FileExistsError('Refusing frozen summary overwrite')
    inventory=json.loads((SOURCE/'completion_inventory.json').read_text())
    for rel,sha in inventory.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
    audit=json.loads((RUN/'compiled/audit.json').read_text());train=read(RUN/'compiled/train.jsonl');ids={r['id'] for r in train}
    assert len(train)==len(ids)==350 and audit['explicit_eos_sites']==150 and audit['development_or_holdout_inputs_read']==0
    assert hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest()==audit['compiled_sha256']
    for arm in ARMS:
        p=RUN/arm;assert (p/'status.txt').read_text().strip()=='COMPLETE'
        m=json.loads((p/'train/train_manifest.json').read_text());a=json.loads((p/'train/terminal_training_audit.json').read_text())
        assert m['train_examples']==a['seen_rows']==350 and a['active_pairs']==a['terminal_sites']==150
        assert a['action_counts']=={'stop':75,'complete_then_stop':75} and a['arm']==arm
        assert a['compiled_data_sha256']==audit['compiled_sha256'] and a['first_action_alpha']==a['terminal_ce_alpha']==.1
        assert m['seed']==20260930 and m['epochs']==1 and m['lr']==1e-5 and m['anchor_alpha']==10
    diagnostic=ROOT/'runs/qwen_terminal_bridge_diagnostic_20261010'
    for _ in range(960):
        state=(diagnostic/'status.txt').read_text().strip()
        if state=='FAILED':raise RuntimeError('Training-site diagnostic failed; inspect log')
        if state=='COMPLETE':break
        time.sleep(15)
    else:raise TimeoutError('Training-site diagnostic did not complete')
    diag=json.loads((diagnostic/'summary.json').read_text())
    assert diag['model_forwards']==1200 and diag['n_training_rows']==150
    assert diag['compiled_sha256']==audit['compiled_sha256'] and diag['development_or_holdout_rows_read']==0
    reachability=ROOT/'runs/qwen_terminal_bridge_reachability_20261010'
    reach=json.loads((reachability/'summary.json').read_text())
    assert reach['compiled_sha256']==audit['compiled_sha256'] and reach['model_generations']==0
    official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    report={'status':'COMPLETE_QWEN_EXPLICIT_TERMINAL_DEVELOPMENT','model':'Qwen3-14B','seed':20260930,
        'new_predictions':480,'holdout_predictions':0,'human_annotations':0,'training_audit':audit,
        'previous_retrieval_round_files_verified_unchanged':len(inventory),'training_site_diagnostic':diag,'training_reachability_audit':reach,'settings':{},'warnings':[
            'Same reused 80+80 internal development questions; not independent confirmation or official leaderboard.',
            'Single training seed; bootstrap estimates question uncertainty, not training-seed variance.',
            'EOS addition increases supervision strength and forward count versus previous round; not an isolated equal-budget ablation.',
            'Terminal states are teacher-forced completed repairs; intermediate completion tokens are not individually supervised.',
            'Q is lexical local citation recall, not semantic entailment, and zero-support questions remain in token/F1 summaries.',
            'Original Qwen often lacks required citation format; comparison cannot equate Q gains with overall factual correctness.',
            'Train/dev zero-support prevalence differs; unknown repetitions remain protected without forced-stop labels.']}
    for setting in ['regression_dev','prospective_dev']:
        p=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
        data={r['id']:r for r in read(p)};assert len(data)==80 and not ids.intersection(data)
        if setting=='regression_dev':
            for k,r in data.items():r['all_gold_answers']=official[k]
        folders={a:RUN/a/setting for a in ARMS}
        folders.update({'previous_'+a:SOURCE/a/setting for a in ARMS+['strong_ce']})
        folders['foundation']=SOURCE/'foundation'/setting
        folders.update({a:ROOT/'runs/submission_20261010/qwen3_frozen_comparison'/setting/a for a in ['baseline','unit_ul']})
        folders['old_clean_pair']=ROOT/'runs/qwen_clean_labels_20261010/action_pair'/setting
        results={};telemetry={};table={}
        for arm,folder in folders.items():
            preds=read(folder/'eval_results.jsonl');tel=read(folder/'termination_telemetry.jsonl')
            assert len(preds)==len(tel)==80 and {r['id'] for r in preds}=={r['id'] for r in tel}==set(data)
            telemetry[arm]={r['id']:r for r in tel}
            assert all(r['tokens']==len(telemetry[arm][r['id']]['generated_token_ids']) for r in preds)
            results[arm]=[score(data[r['id']],r) for r in preds];table[arm]=summarize(results[arm])
            table[arm]['true_cap_hit_rate']=float(np.mean([r['hit_cap_without_eos'] for r in tel]))
        pairs=[(a,b) for a in ARMS for b in ['baseline','previous_'+a,'foundation']]
        pairs += [('action_pair',b) for b in ['action_ce','action_ul','previous_strong_ce','unit_ul','old_clean_pair']]
        comparisons={a+'_minus_'+b:paired_fast(results[a],results[b],telemetry[a],telemetry[b]) for a,b in pairs}
        strata={}
        for label,nonzero in [('nonzero_lexical_support',True),('zero_lexical_support',False)]:
            selected={k for k,r in data.items() if bool(valid_training_edges(r['edges']))==nonzero}
            if not selected:continue
            sub={a:[r for r in rows if r['id'] in selected] for a,rows in results.items()}
            strata[label]={'n':len(selected),'table':{a:summarize(rows) for a,rows in sub.items()},
                'comparisons':{a+'_minus_baseline':paired_fast(sub[a],sub['baseline'],telemetry[a],telemetry['baseline']) for a in ARMS}}
            for a,rows in sub.items():strata[label]['table'][a]['true_cap_hit_rate']=float(np.mean([telemetry[a][r['id']]['hit_cap_without_eos'] for r in rows]))
            if not nonzero:
                reason='Zero constructed lexical edges: Q/U and Q noninferiority undefined.';strata[label]['warning']=reason
                for t in strata[label]['table'].values():t['grounded_quality']=t['utility']=None
                for c in strata[label]['comparisons'].values():
                    for k in ['grounded_quality','utility','relative_utility','quality_noninferiority']:c[k]={'status':'UNDEFINED_FOR_ZERO_EDGE_STRATUM','reason':reason}
        report['settings'][setting]={'n':80,'data_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'table':table,
            'comparisons':comparisons,'input_support_strata':strata}
    report['development_gates']={}
    for arm in ARMS:
        v=[s['comparisons'][arm+'_minus_baseline'] for s in report['settings'].values()]
        g={'quality_noninferior_both':all(c['quality_noninferiority']['supported'] for c in v),
            'utility_ci_positive_both':all(c['relative_utility']['ci95'][0]>0 for c in v),
            'no_clear_f1_loss_both':all(c['answer_f1']['ci95'][1]>=0 for c in v),
            'no_clear_token_or_cap_regression_both':all(c['mean_tokens']['ci95'][0]<=0 and c['true_cap_hit_rate']['ci95'][0]<=0 for c in v)}
        g['ready_for_confirmatory_expansion']=all(g.values());report['development_gates'][arm]=g
    checkpoint_hashes={}
    folders=[ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter']
    folders += [base/a/phase/'adapter' for base,arms in [(RUN,ARMS),(SOURCE,ARMS+['strong_ce'])] for a in arms for phase in ['train','interp05']]
    for folder in folders:
        for name in ['adapter_model.safetensors','adapter_config.json']:
            p=folder/name;assert p.exists(),str(p)
            checkpoint_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RUN/'checkpoint_hashes.json').write_text(json.dumps(checkpoint_hashes,indent=2))
    report['checkpoint_files_hashed_without_weight_download']=len(checkpoint_hashes)
    target.write_text(json.dumps(report,indent=2))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*') if p.is_file() and 'adapter' not in p.parts and p.name!='completion_inventory.json'}
    files.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in diagnostic.rglob('*') if p.is_file()})
    files.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in reachability.rglob('*') if p.is_file()})
    for p in (ROOT/'code').glob('*qwen_terminal_bridge*20261010*'):
        if p.is_file():files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'logs/qwen_terminal_bridge_20261010_protocol.md';files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'logs/qwen_terminal_bridge_20261010/diagnostic_protocol.md';files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RUN/'completion_inventory.json').write_text(json.dumps(files,indent=2))
    members=set(files)|{str((RUN/'completion_inventory.json').relative_to(ROOT))}
    members.update(str(p.relative_to(ROOT)) for p in (ROOT/'logs/qwen_terminal_bridge_20261010').rglob('*') if p.is_file())
    (ROOT/'exports/qwen_terminal_bridge_20261010_members.list').write_text('\n'.join(sorted(members))+'\n')
    print(json.dumps({'tables':{k:s['table'] for k,s in report['settings'].items()},'gates':report['development_gates']},indent=2),flush=True)

if __name__=='__main__':main()

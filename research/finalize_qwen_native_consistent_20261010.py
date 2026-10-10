"""Frozen native-prefix pilot, matched controls and paired development statistics."""
import hashlib,json
import numpy as np
from compile_qwen_native_consistent_20261010 import ROOT,RUN,SOURCE
from completion_pairs_20261010 import valid_training_edges
from finalize_qwen_priority_20261010 import read,paired_fast
from score_strict_units_20261010 import score,summarize
ARMS=['action_pair','action_ce','action_ul']
RETRIEVAL=ROOT/'runs/qwen_retrieval_align_20261010'

def main():
    target=RUN/'development_summary.json'
    if target.exists():raise FileExistsError('Refusing summary overwrite')
    inventory=json.loads((SOURCE/'completion_inventory.json').read_text())
    for rel,sha in inventory.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
    audit=json.loads((RUN/'compiled/audit.json').read_text());train=read(RUN/'compiled/train.jsonl');ids={r['id'] for r in train}
    assert len(train)==len(ids)==350 and audit['native_eos_sites']==72 and audit['explicit_eos_sites']==150 and not audit['exact_token_path_conflicts']
    assert hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest()==audit['compiled_sha256']
    for arm in ARMS:
        p=RUN/arm;assert (p/'status.txt').read_text().strip()=='COMPLETE'
        m=json.loads((p/'train/train_manifest.json').read_text());a=json.loads((p/'train/terminal_training_audit.json').read_text());n=json.loads((p/'train/native_training_manifest.json').read_text())
        assert m['train_examples']==a['seen_rows']==n['seen_rows']==350 and a['active_pairs']==a['terminal_sites']==150
        assert a['action_counts']=={'stop':75,'complete_then_stop':75} and a['arm']==n['arm']==arm
        assert n['native_eos_sites']==72 and n['same_loss_weights_and_forward_budget_as_bridge']
        assert a['compiled_data_sha256']==audit['compiled_sha256'] and a['first_action_alpha']==a['terminal_ce_alpha']==.1
        assert m['seed']==20260930 and m['epochs']==1 and m['lr']==1e-5 and m['anchor_alpha']==10
    official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    report={'status':'COMPLETE_QWEN_NATIVE_CONSISTENT_DEVELOPMENT','model':'Qwen3-14B','seed':20260930,
        'new_predictions':480,'holdout_predictions':0,'human_annotations':0,'training_audit':audit,
        'previous_bridge_files_verified_unchanged':len(inventory),'settings':{},'warnings':[
            'Same repeatedly reused 80+80 development questions; not official leaderboard or independent confirmation.',
            'Single training seed; question bootstrap does not estimate seed variation.',
            'Jointly repaired first-action and terminal supervision on 72 stops; 78 counterfactuals unchanged; four decoded-only prefix flags remain, no exact conflict.',
            'Token-alignment repair is an implementation prerequisite, not sufficient method novelty.',
            'Q is lexical local citation recall, not semantic entailment. Unknown zero-edge repetitions retain teacher protection.',
            'Finite candidate grammar, sparse first-action supervision and soft KL do not guarantee complete factual answers.',
            'Train/raw-dev zero-edge prevalence differs; original Qwen citation-format failures limit interpretation of its Q.']}
    for setting in ['regression_dev','prospective_dev']:
        p=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
        data={r['id']:r for r in read(p)};assert len(data)==80 and not ids.intersection(data)
        if setting=='regression_dev':
            for k,r in data.items():r['all_gold_answers']=official[k]
        folders={a:RUN/a/setting for a in ARMS}
        folders.update({'bridge_'+a:SOURCE/a/setting for a in ARMS})
        folders.update({'retrieval_'+a:RETRIEVAL/a/setting for a in ARMS})
        folders['foundation']=RETRIEVAL/'foundation'/setting;folders['strong_ce']=RETRIEVAL/'strong_ce'/setting
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
        pairs=[(a,b) for a in ARMS for b in ['baseline','bridge_'+a,'retrieval_'+a,'foundation']]
        pairs += [('action_pair',b) for b in ['action_ce','action_ul','strong_ce','unit_ul','old_clean_pair']]
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
                reason='Zero lexical edges: Q/U/quality noninferiority undefined.';strata[label]['warning']=reason
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
    hashes={}
    for folder in [RUN/a/phase/'adapter' for a in ARMS for phase in ['train','interp05']]:
        for name in ['adapter_model.safetensors','adapter_config.json']:
            p=folder/name;assert p.exists();hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RUN/'checkpoint_hashes.json').write_text(json.dumps(hashes,indent=2));target.write_text(json.dumps(report,indent=2))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*') if p.is_file() and 'adapter' not in p.parts and p.name!='completion_inventory.json'}
    for p in (ROOT/'code').glob('*qwen_native_terminal*20261010*'):
        if p.is_file():files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'logs/qwen_native_consistent_20261010_protocol.md';files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RUN/'completion_inventory.json').write_text(json.dumps(files,indent=2));members=set(files)|{str((RUN/'completion_inventory.json').relative_to(ROOT))}
    members.update(str(p.relative_to(ROOT)) for p in (ROOT/'logs/qwen_native_consistent_20261010').rglob('*') if p.is_file())
    (ROOT/'exports/qwen_native_consistent_20261010_members.list').write_text('\n'.join(sorted(members))+'\n')
    print(json.dumps({'tables':{k:s['table'] for k,s in report['settings'].items()},'gates':report['development_gates']},indent=2),flush=True)

if __name__=='__main__':main()

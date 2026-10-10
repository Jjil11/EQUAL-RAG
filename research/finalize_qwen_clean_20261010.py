"""Full development evidence for clean targets, retaining all previous controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from qwen_clean_labels_20261010 import ROOT,RUN
from completion_pairs_20261010 import valid_training_edges,covered,norm
from score_strict_units_20261010 import score,summarize
from finalize_qwen_priority_20261010 import read,paired_fast

ARMS=['seq_pair','seq_sft','action_pair','action_ce']
CONTROLS=['action_ul','strong_ce']


def main():
    target=RUN/'development_summary.json'
    if target.exists():raise FileExistsError('Refusing frozen summary overwrite')
    prior=ROOT/'runs/qwen_priority_20261010';older=ROOT/'runs/completion_repair_20261010';old=ROOT/'runs/submission_20261010'
    inventory=json.loads((prior/'completion_inventory.json').read_text())
    for rel,expected in inventory.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
    train=read(RUN/'compiled/train.jsonl');ids={r['id'] for r in train};assert len(ids)==len(train)==350
    audit=json.loads((RUN/'compiled/audit.json').read_text())
    assert audit['compiled_sha256']==hashlib.sha256((RUN/'compiled/train.jsonl').read_bytes()).hexdigest()
    assert (RUN/'diagnostic_status.txt').read_text().strip()=='COMPLETE'
    diagnostic=json.loads((RUN/'action_diagnostic_summary.json').read_text())
    for arm in ARMS:
        p=RUN/arm;assert (p/'status.txt').read_text().strip()=='COMPLETE'
        manifest=json.loads((p/'train/train_manifest.json').read_text());a=json.loads((p/'train/clean_training_audit.json').read_text())
        assert manifest['train_examples']==a['seen_rows']==350 and a['active_pairs']==39
        assert manifest['seed']==20260930 and manifest['epochs']==1 and manifest['trajectory_alpha']==.1
        assert a['action_counts']=={'stop':2,'complete_then_stop':37}
    controls=ROOT/'runs/qwen_clean_controls_20261010'
    for arm in CONTROLS:
        p=controls/arm;assert (p/'status.txt').read_text().strip()=='COMPLETE'
        manifest=json.loads((p/'train/train_manifest.json').read_text())
        a=json.loads((p/'train/clean_training_audit.json').read_text())
        c=json.loads((p/'train/supplemental_control_manifest.json').read_text())
        assert manifest['train_examples']==a['seen_rows']==350 and a['active_pairs']==39
        assert a['action_counts']=={'stop':2,'complete_then_stop':37}
        assert c['control']==arm and c['repair_alpha']==(.2 if arm=='strong_ce' else .1)
    official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in
              enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    report={'status':'COMPLETE_CLEAN_TARGET_QWEN_DEVELOPMENT','seed':20260930,'model':'Qwen3-14B',
        'holdout_predictions':0,'human_annotations':0,'new_predictions':960,'training_audit':audit,
        'training_context_action_diagnostic':diagnostic,
        'priority_round_files_verified_unchanged':len(inventory),'settings':{},'warnings':[
            'Reused small development sets, multiple comparisons unadjusted; not independent confirmation.',
            'Lexical local citation proxy is not entailment or human-grounded truth.',
            'Citation corrections advance four training forks; this is not exclusively a repetition-boundary intervention.',
            'All aliases/candidates are bounded, not an exhaustive correct language set.',
            'Sequence preference adds branch gradients; action arms are forward-matched but not gradient-norm matched.',
            'Single seed confidence intervals reflect question sampling, not training randomness.',
            'Baseline is the frozen trained A-EQUAL anchor, not the unadapted foundation model.']}
    retrieval_audit=ROOT/'runs/qwen_training_retrieval_audit_v2_20261010/summary.json'
    report['train_retrieval_condition_audit']=json.loads(retrieval_audit.read_text())
    assert report['train_retrieval_condition_audit']['status']=='COMPLETE_CORRECTED_TRAIN_ONLY_AUDIT'
    for setting in ['regression_dev','prospective_dev']:
        source=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
        data={r['id']:r for r in read(source)};assert len(data)==80 and not ids.intersection(data)
        if setting=='regression_dev':
            for k,r in data.items():r['all_gold_answers']=official[k]
        folders={a:old/'qwen3_frozen_comparison'/setting/a for a in ['baseline','unit_ul','semantic_eos']}
        folders.update({a:older/('qwen3_'+a.removeprefix('previous_'))/setting for a in ['previous_pair','previous_sft']})
        folders.update({'priority_'+a:prior/a/setting for a in ['seq_full','seq_local','action_pair','action_ce']})
        folders.update({a:RUN/a/setting for a in ARMS})
        folders.update({a:controls/a/setting for a in CONTROLS})
        results={};tel={};table={}
        for a,p in folders.items():
            preds=read(p/'eval_results.jsonl');tokens=read(p/'termination_telemetry.jsonl')
            assert len(preds)==len(tokens)==80 and {r['id'] for r in preds}==set(data)
            tel[a]={r['id']:r for r in tokens};assert len(tel[a])==len({r['id'] for r in preds})==80
            assert all(r['tokens']==len(tel[a][r['id']]['generated_token_ids']) for r in preds)
            results[a]=[score(data[r['id']],r) for r in preds];table[a]=summarize(results[a])
            table[a]['true_cap_hit_rate']=float(np.mean([r['hit_cap_without_eos'] for r in tokens]))
        pairs=[(a,b) for a in ARMS for b in ['baseline','unit_ul','previous_pair']]
        pairs += [('seq_pair','seq_sft'),('action_pair','action_ce'),('action_pair','seq_pair'),
                  ('seq_pair','priority_seq_local'),('action_pair','priority_action_pair'),('action_ce','priority_action_ce')]
        pairs += [(a,b) for a in ['action_pair','seq_pair'] for b in CONTROLS]
        pairs += [(a,'baseline') for a in CONTROLS]
        comparisons={a+'_minus_'+b:paired_fast(results[a],results[b],tel[a],tel[b]) for a,b in pairs}
        robust={}
        for arm,rows in results.items():
            rescored=[]
            for pred in rows:
                clean=dict(data[pred['id']]);clean['edges']=valid_training_edges(clean['edges'])
                clean['all_gold_answers']=[g for g in clean['all_gold_answers'] if any(norm(a) for a in g)]
                item=score(clean,dict(pred,legacy_grounded_edges=pred['legacy_grounded_edges']))
                item['grounded_edges']=len(covered(pred['output'],clean['edges']));rescored.append(item)
            robust[arm]=rescored
        strata={}
        for label,has in [('nonzero_lexical_support',True),('zero_lexical_support',False)]:
            selected={k for k,r in data.items() if bool(valid_training_edges(r['edges']))==has}
            if selected:strata[label]={'n':len(selected),'table':{a:summarize([r for r in rows if r['id'] in selected]) for a,rows in results.items()}}
        report['settings'][setting]={'n':80,'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'table':table,'comparisons':comparisons,'input_support_strata':strata,
            'supplemental_clean_citation_group_parser':{
                'warning':'Sensitivity analysis applied uniformly to all arms; primary metric is unchanged.',
                'table':{a:summarize(r) for a,r in robust.items()},
                'comparisons':{a+'_minus_'+b:paired_fast(robust[a],robust[b],tel[a],tel[b]) for a,b in
                    [('action_pair','baseline'),('action_pair','action_ce'),('seq_pair','baseline'),('seq_pair','seq_sft')]}}}
    report['development_gates']={}
    for a in ARMS:
        values=[s['comparisons'][a+'_minus_baseline'] for s in report['settings'].values()]
        gate={'quality_noninferior_both':all(c['quality_noninferiority']['supported'] for c in values),
              'utility_ci_positive_both':all(c['relative_utility']['ci95'][0]>0 for c in values),
              'no_clear_f1_loss_both':all(c['answer_f1']['ci95'][1]>=0 for c in values),
              'no_clear_token_or_cap_regression_both':all(c['mean_tokens']['ci95'][0]<=0 and c['true_cap_hit_rate']['ci95'][0]<=0 for c in values)}
        gate['ready_for_confirmatory_expansion']=all(gate.values());report['development_gates'][a]=gate
    target.write_text(json.dumps(report,indent=2))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*')
           if p.is_file() and 'adapter' not in p.parts and p.name!='completion_inventory.json'}
    files.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in controls.rglob('*')
                  if p.is_file() and 'adapter' not in p.parts})
    for pattern in ['*qwen_clean*20261010*']:
        for p in (ROOT/'code').glob(pattern):
            if p.is_file():files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'logs/qwen_clean_protocol_20261010.md';files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'logs/qwen_clean_supplemental_protocol_20261010.md';files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [retrieval_audit,ROOT/'code/audit_qwen_training_retrieval_v2_20261010.py']:
        files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RUN/'completion_inventory.json').write_text(json.dumps(files,indent=2))
    # Derived file list prevents omission of newly added code/inventory entries.
    members=set(files)|{str((RUN/'completion_inventory.json').relative_to(ROOT))}
    for folder in ['logs/qwen_clean_labels_20261010','logs/qwen_clean_controls_20261010']:
        members.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file())
    (ROOT/'exports/qwen_clean_labels_20261010_members.list').write_text('\n'.join(sorted(members))+'\n')
    print(json.dumps({'tables':{k:s['table'] for k,s in report['settings'].items()},'gates':report['development_gates']},indent=2),flush=True)


if __name__=='__main__':main()

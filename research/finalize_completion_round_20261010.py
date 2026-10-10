"""Development-only full-gold, paired, termination and data-integrity report."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
from pathlib import Path
import numpy as np
from score_strict_units_20261010 import score,summarize,paired,norm
from finalize_verified_all_answers_20261010 import relative_utility
from completion_pairs_20261010 import valid_training_edges,covered,true_repeat

ROOT=Path(str(_release_root))
RUN=ROOT/'runs/completion_repair_20261010'
OLD=ROOT/'runs/submission_20261010'


def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    report={'status':'COMPLETE_DEVELOPMENT_PILOT','training_seed':20260930,'holdout_predictions':0,
            'human_annotations':0,'families':{},'warnings':[
            'Development comparisons only; no multiplicity adjustment or independent confirmation.',
            'Grounded quality is a lexical local citation proxy, not entailment.',
            'Four arms share new labels and KL mask; they do not isolate KL or label revisions individually.',
            'Candidate preference term adds optimization signal; same-label SFT is not a gradient-norm-matched control.',
            'No guarantee that repaired suffix can retract incorrect units already emitted.']}
    official={str(x.get('id',x.get('sample_id',i))):x['answers'] for i,x in enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
    old_hashes=json.loads((OLD/'completion_inventory.json').read_text())['file_sha256']
    preserved=0
    for rel,expected in old_hashes.items():
        if rel.startswith('code/') or rel.startswith('data/') or '_frozen_comparison/' in rel or rel.startswith('ALCE/'):
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected, f'Historical artifact changed: {rel}'
            preserved+=1
    report['historical_files_hash_verified_unchanged']=preserved
    for family in ['qwen3','mistral']:
        train=read(RUN/'compiled_v3'/f'{family}_train.jsonl')
        train_ids={x['id'] for x in train};assert len(train_ids)==len(train)==350
        orig_file='train_trajectory_qwen3_14b_onpolicy_seed20260930.jsonl' if family=='qwen3' else 'train_trajectory_onpolicy_seed20260930.jsonl'
        original={x['id']:x for x in read(ROOT/'data/qampari_evidence_full'/orig_file)}
        audit=json.loads((RUN/'compiled_v3'/f'{family}_audit.json').read_text())
        assert audit['compiled_sha256']==hashlib.sha256((RUN/'compiled_v3'/f'{family}_train.jsonl').read_bytes()).hexdigest()
        relocated=sum(x.get('has_trajectory') and x['trajectory_prefix']!=original[x['id']]['trajectory_prefix'] for x in train)
        report['families'][family]={'training_audit':{k:v for k,v in audit.items() if k!='per_row'},
                                  'relocated_text_boundaries':int(relocated),'settings':{}}
        report['families'][family]['mechanism_diagnostic']=json.loads((RUN/'mechanism_diagnostic'/f'{family}_summary.json').read_text())
        report['families'][family]['training_prefix_replay']=json.loads((RUN/'prefix_replay'/f'{family}_summary.json').read_text())
        for arm in ['pair','sft']:
            folder=RUN/f'{family}_{arm}'
            assert (folder/'status.txt').read_text().strip()=='COMPLETE'
            manifest=json.loads((folder/'train/train_manifest.json').read_text())
            assert manifest['seed']==20260930 and manifest['epochs']==1 and manifest['train_examples']==350
            assert manifest['lr']==1e-5 and manifest['trajectory_alpha']==.1 and manifest['anchor_alpha']==10
            a=json.loads((folder/'train/completion_training_audit.json').read_text())
            assert a['seen_rows']==350 and a['active_pairs']==audit['active_pairs']
            assert a['action_counts']=={'stop':audit['stop_pairs'],'complete_then_stop':audit['completion_pairs']}
        for setting in ['regression_dev','prospective_dev']:
            datafile=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
            data={x['id']:x for x in read(datafile)};assert len(data)==80 and not train_ids.intersection(data)
            if setting=='regression_dev':
                for k,row in data.items():row['all_gold_answers']=official[k]
            results={};telemetry={};robust={}
            for arm in ['baseline','unit_ul','semantic_eos','sft','pair']:
                folder=RUN/f'{family}_{arm}'/setting if arm in ['sft','pair'] else OLD/f'{family}_frozen_comparison'/setting/arm
                preds=read(folder/'eval_results.jsonl');tel=read(folder/'termination_telemetry.jsonl')
                assert len(preds)==len(tel)==80 and {x['id'] for x in preds}==set(data)
                assert len({x['id'] for x in preds})==80 and len({x['id'] for x in tel})==80
                telemetry[arm]={x['id']:x for x in tel}
                assert all(p['tokens']==len(telemetry[arm][p['id']]['generated_token_ids']) for p in preds)
                results[arm]=[score(data[x['id']],x) for x in preds]
                r=[]
                for x in preds:
                    clean=valid_training_edges(data[x['id']]['edges'])
                    cleanrow=dict(data[x['id']]);cleanrow['edges']=clean
                    cleanrow['all_gold_answers']=[g for g in data[x['id']]['all_gold_answers'] if any(norm(a) for a in g)]
                    item=score(cleanrow,x)
                    item['gold_edges']=len(clean);item['grounded_edges']=len(covered(x['output'],clean))
                    r.append(item)
                robust[arm]=r
            table={arm:summarize(rows) for arm,rows in results.items()}
            for arm,rows in results.items():
                table[arm]['true_cap_hit_rate']=float(np.mean([telemetry[arm][x['id']]['hit_cap_without_eos'] for x in rows]))
                table[arm]['token_quantiles']=dict(zip(['p50','p90','p95','p99'],np.quantile([x['tokens'] for x in rows],[.5,.9,.95,.99]).tolist()))
            comparisons={}
            for arm,base in [('pair','baseline'),('pair','sft'),('pair','unit_ul'),('sft','baseline')]:
                c=paired(results[arm],results[base]);c['relative_utility']=relative_utility(results[arm],results[base])
                # True cap events, not a fixed-length proxy.
                a=np.array([telemetry[arm][x['id']]['hit_cap_without_eos'] for x in results[arm]],float)
                # Sort for correct per-question pairing independently of file order.
                ids=sorted(data);a=np.array([telemetry[arm][i]['hit_cap_without_eos'] for i in ids],float)
                b=np.array([telemetry[base][i]['hit_cap_without_eos'] for i in ids],float)
                ix=np.random.default_rng(20261010).integers(0,80,(20000,80))
                c['true_cap_hit_rate']={'delta':float((a-b).mean()),'ci95':np.quantile((a-b)[ix].mean(1),[.025,.975]).tolist()}
                comparisons[f'{arm}_minus_{base}']=c
            strata={}
            for name,ids in [('nonzero_lexical_support',{i for i,r in data.items() if valid_training_edges(r['edges'])}),
                             ('zero_lexical_support',{i for i,r in data.items() if not valid_training_edges(r['edges'])})]:
                if not ids:continue
                selected={arm:[x for x in rows if x['id'] in ids] for arm,rows in results.items()}
                strata[name]={'n':len(ids),'warning':'Frozen input-derived diagnostic strata, not inference routing or confirmation of semantic evidence absence.',
                              'table':{arm:summarize(rows) for arm,rows in selected.items()},
                              'pair_minus_baseline':paired(selected['pair'],selected['baseline']),
                              'pair_minus_sft':paired(selected['pair'],selected['sft'])}
            report['families'][family]['settings'][setting]={
                'n':80,'data_sha256':hashlib.sha256(datafile.read_bytes()).hexdigest(),
                'table':table,'comparisons':comparisons,'input_support_strata':strata,
                'sensitivity_citation_group_parser':{
                    'warning':'Supplemental re-score of ALL arms, excluding empty gold edges/answer groups; citation-group parsing only changes grounding. Official comma-list F1 and original proxy remain primary.',
                    'invalid_empty_edges':sum(len(x['edges'])-len(valid_training_edges(x['edges'])) for x in data.values()),
                    'table':{a:summarize(r) for a,r in robust.items()},
                    'pair_minus_sft':paired(robust['pair'],robust['sft']),
                    'pair_minus_baseline':paired(robust['pair'],robust['baseline'])}}
        settings=report['families'][family]['settings']
        quality=all(v['comparisons']['pair_minus_baseline']['quality_noninferiority']['supported'] for v in settings.values())
        efficiency=all(v['comparisons']['pair_minus_baseline']['relative_utility']['ci95'][0]>0 for v in settings.values())
        extra=any(v['comparisons']['pair_minus_sft']['relative_utility']['ci95'][0]>0 for v in settings.values())
        f1_no_clear_loss=all(v['comparisons']['pair_minus_baseline']['answer_f1']['ci95'][1]>=0 for v in settings.values())
        report['families'][family]['development_gate']={'quality_proxy_noninferior_both':quality,
            'utility_gain_ci_above_zero_both':efficiency,'pair_extra_utility_ci_above_zero_any':extra,
            'no_clear_f1_loss_both':f1_no_clear_loss,
            'sufficient_for_immediate_pair_three_seed_expansion':quality and efficiency and extra and f1_no_clear_loss,
            'warning':'Conservative budget gate on two small, reused development sets; not an acceptance prediction or proof of no benefit when false.'}
    target=RUN/'development_summary.json'
    if target.exists():raise FileExistsError('Refusing to overwrite final report')
    target.write_text(json.dumps(report,indent=2))
    inventory={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*')
               if p.is_file() and 'adapter' not in p.parts and p.name!='completion_inventory.json'}
    for p in (ROOT/'code').glob('*completion*20261010*'):
        if p.is_file():inventory[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'code/audit_evidence_decisions_20261010.py'
    inventory[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    p=ROOT/'code/replay_evidence_prefixes_20261010.py'
    inventory[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for rel in ['code/train_trajectory_equal.py','code/train_verified_boundary_20261010.py',
                'code/eval_verified_outputs_20261010.py','code/score_strict_units_20261010.py',
                'code/finalize_verified_all_answers_20261010.py','code/interpolate_peft_adapters.py',
                'ALCE/data/qampari_eval_gtr_top100.json','data/qampari_evidence_full/dev.jsonl',
                'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl']:
        inventory[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    for family in ['qwen3','mistral']:
        for setting in ['regression_dev','prospective_dev']:
            for arm in ['baseline','unit_ul','semantic_eos']:
                for name in ['eval_results.jsonl','termination_telemetry.jsonl']:
                    p=OLD/f'{family}_frozen_comparison'/setting/arm/name
                    inventory[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (RUN/'completion_inventory.json').write_text(json.dumps(inventory,indent=2))
    print(json.dumps({family:{setting:{arm:{k:v for k,v in table.items() if k in ['grounded_quality','mean_tokens','answer_f1','true_cap_hit_rate','utility']} for arm,table in item['table'].items()}
                             for setting,item in report['families'][family]['settings'].items()} for family in report['families']},indent=2),flush=True)


if __name__=='__main__':main()

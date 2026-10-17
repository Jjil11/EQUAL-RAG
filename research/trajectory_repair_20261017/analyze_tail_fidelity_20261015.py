"""Frozen development analysis and name-group ledger, not model selection."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import re
from collections import Counter
import qwen_tail_fidelity_20261015 as c
from score_strict_units_20261010 import score,summarize,norm
from analyze_qwen_base_entity_abc_20261015 import paired_fixed

q=c.q;RUN=c.RUN


def membership(row,pred,cited):
    units=[u.strip() for u in pred['output'].rstrip().rstrip('.').rstrip(',').split(',') if norm(u)]
    hits=set()
    for i,g in enumerate(row['edges']):
        aliases={norm(a) for a in g['aliases'] if norm(a)}
        supports=set(g.get('supporting_doc_ids',[g['doc_id']]))
        if any(norm(u) in aliases and (not cited or supports&{int(v) for v in re.findall(r'\[(\d+)\]',u)}) for u in units):hits.add(i)
    return hits


def name_audit(predictions,inputs,folder):
    old={p['id']:p for p in q.read(c.PREVIOUS/'eval/base_pair/predictions.jsonl')}
    output=[]
    for key,row in inputs.items():
        if row['eval_split']!='multi45':continue
        item={'id':key,'edges':row['edges']}
        for kind,cited in [('name_only',False),('locally_cited',True)]:
            base=membership(row,predictions['base'][key],cited)
            pair=membership(row,predictions['base_pair'][key],cited)
            before=membership(row,old[key],cited)
            item[kind]={'base':sorted(base),'pair':sorted(pair),'added':sorted(pair-base),'lost':sorted(base-pair),
                'added_names':[row['edges'][i]['aliases'] for i in sorted(pair-base)],
                'lost_names':[row['edges'][i]['aliases'] for i in sorted(base-pair)],
                'old_added_retained':sorted((before-base)&pair),
                'old_added_no_longer_hit':sorted((before-base)-pair),
                'old_lost_recovered':sorted((base-before)&pair),
                'old_lost_still_missing':sorted((base-before)-pair)}
        base=membership(row,predictions['base'][key],False);pair=membership(row,predictions['base_pair'][key],False)
        bc=membership(row,predictions['base'][key],True);pc=membership(row,predictions['base_pair'][key],True)
        item['Q_decomposition']={'new_name_with_supported_citation':sorted(pc-base),
            'old_name_citation_repaired':sorted((pc&base)-bc),
            'original_supported_name_missing':sorted(bc-pair),
            'name_retained_but_supported_citation_lost':sorted((bc&pair)-pc)}
        output.append(item)
    q.write_rows(folder/'name_groups45.jsonl',output)
    result={kind:{field:sum(len(r[kind][field]) for r in output) for field in
        ['base','pair','added','lost','old_added_retained','old_added_no_longer_hit','old_lost_recovered','old_lost_still_missing']} for kind in ['name_only','locally_cited']}
    for kind in result:
        result[kind]['net']=result[kind]['added']-result[kind]['lost']
    result['Q_decomposition']={field:sum(len(r['Q_decomposition'][field]) for r in output) for field in output[0]['Q_decomposition']}
    g=result['locally_cited']
    result['requested_count_goal_met']=bool(g['lost']<=3 and g['added']>=17)
    result['requested_previous_17_additions_all_retained']=g['old_added_retained']==17
    result['positive_net_Q']=g['net']>0
    result['not_training_checkpoint_selection']=True
    q.dump(folder/'name_groups45_summary.json',result);return result


def old_cause_audit():
    """Evidence compatible with, not proof of, a simple truncation cause."""
    inputs={r['id']:r for r in q.read(RUN/'eval69.jsonl') if r['eval_split']=='multi45'}
    base={p['id']:p for p in q.read(RUN/'cached_eval/base.jsonl')}
    old={p['id']:p for p in q.read(c.PREVIOUS/'eval/base_pair/predictions.jsonl')}
    counts=Counter();rows=[]
    for key,row in inputs.items():
        a,b=old[key],base[key];pa=a['generated_token_ids'];pb=b['generated_token_ids']
        common=c.common_prefix(pa,pb)
        names=lambda p:[norm(u) for u in p['output'].rstrip().rstrip('.').rstrip(',').split(',') if norm(u)]
        an,bn=names(a),names(b)
        prefix=bool(len(an)<len(bn) and an==bn[:len(an)] and a['ended_by_eos'])
        loss=membership(row,b,True)-membership(row,a,True)
        counts['questions_first_generated_token_changed']+=common==0
        counts['list_prefix_truncation_questions']+=prefix
        counts['lost_Q_groups_in_list_prefix_truncation']+=len(loss) if prefix else 0
        counts['lost_Q_groups_other_trajectory_changes']+=len(loss) if not prefix else 0
        rows.append({'id':key,'token_common_prefix':common,'list_prefix_truncation_compatible':prefix,'lost_Q_groups':len(loss)})
    return {'counts':dict(counts),'rows':rows,'causal_C_ablation_not_performed':True,
        'historical_C_not_assumed_Base_native':True}


def main():
    inputs=q.read(RUN/'eval69.jsonl');byid={r['id']:r for r in inputs}
    folder=RUN/'final_analysis';folder.mkdir()
    models=['base','base_block','anchor','base_ce','base_pair'];predictions={};scored={};tables={}
    for model in models:
        path=RUN/'cached_eval'/(model+'.jsonl') if model in ['base','anchor'] else RUN/'eval'/model/'predictions.jsonl'
        ps=q.read(path);assert len(ps)==69 and len({p['id'] for p in ps})==69
        assert {p['id'] for p in ps}==set(byid)
        predictions[model]={p['id']:p for p in ps}
        if model in ['base_ce','base_pair']:
            assert all(not p['blocking_events'] for p in ps)
            assert json.loads((RUN/model/'gate_latest.json').read_text())['passed']
            assert all(p['tokens']==len(p['generated_token_ids']) for p in ps)
        for p in ps:
            assert p['prompt_token_ids']==predictions['base'][p['id']]['prompt_token_ids']
            assert p['hit_cap_without_eos']==(len(p['generated_token_ids'])==256 and p['generated_token_ids'][-1] not in c.STOPS)
        values=[dict(score(byid[p['id']],dict(p,grounded_edges=0)),cap_hit_proxy=p['hit_cap_without_eos']) for p in ps]
        q.write_rows(folder/(model+'_scored69.jsonl'),values);scored[model]={r['id']:r for r in values};tables[model]={}
        for split in ['multi45','raw24']:
            selected=[r for r in values if byid[r['id']]['eval_split']==split]
            tables[model][split]=dict(summarize(selected),true_cap_hit_rate=sum(r['cap_hit_proxy'] for r in selected)/len(selected))
    intervals={}
    comparisons=[(m,'base') for m in models if m!='base']+[('base_pair','base_ce'),('base_pair','base_block')]
    for model,baseline in comparisons:
        for split in ['multi45','raw24']:
            ids=sorted(k for k,r in byid.items() if r['eval_split']==split)
            intervals[model+'__minus__'+baseline+'__'+split]=paired_fixed([scored[model][k] for k in ids],[scored[baseline][k] for k in ids],samples=20000)
    manifests={m:json.loads((RUN/m/'manifest.json').read_text()) for m in ['base_ce','base_pair']}
    for key in ['initial_trainable_weights_sha256','data_sha256','retention_sha256','trainable_parameter_count','script_sha256','driver_sha256']:
        assert manifests['base_ce'][key]==manifests['base_pair'][key],key
    assert all(not m['old_Anchor_loaded'] for m in manifests.values())
    result={'tables':tables,'paired_bootstrap':intervals,'bootstrap_samples':20000,'seed':20261010,
        'gates':{m:json.loads((RUN/m/'gate_latest.json').read_text()) for m in manifests},
        'training_base_diagnostic':json.loads((RUN/'diagnostic/base/summary.json').read_text()),
        'actual_budgets':{m:{k:manifests[m][k] for k in ['updates','supervised_visits','aux_retention_visits']} for m in manifests},
        'same_update_budget':manifests['base_ce']['updates']==manifests['base_pair']['updates'],
        'independent_early_stopping_can_confound_loss_only_comparison':True,
        'same_initial_weights_data_and_maximum_budget':True,'main_models_use_decoding_rules':False,
        'reused_dev_not_blind':True,'semantic_support_certified':False,'sealed311_inputs_opened':0,
        'script_sha256':q.sha(__file__)}
    names=name_audit(predictions,byid,folder)
    assert names['locally_cited']['base']==round(tables['base']['multi45']['grounded_quality']*176)
    assert names['locally_cited']['pair']==round(tables['base_pair']['multi45']['grounded_quality']*176)
    result['name_group_audit']=names
    records={r['id']:r for r in q.read(RUN/'compiled/records.jsonl')}
    detail={}
    for model in ['base','base_ce','base_pair']:
        ps=q.read(RUN/'diagnostic'/model/'predictions.jsonl')
        selected=[p for p in ps if p['type']=='A']
        detail[model]={'A_records':len(selected),
            'A_unique_native_prefixes':len({records[p['id']]['native_id'] for p in selected}),
            'A_unique_prefix_valid_success':len({records[p['id']]['native_id'] for p in selected if p['valid_type_success']}),
            'A_first_native_continuation_token':sum(bool(p['generated_ids']) and p['generated_ids'][0]==1125 for p in selected),
            'A_exact_first_target':sum(p['conditional_prefix_success'] for p in selected),
            'A_complete_chain':sum(p['complete_chain_success'] for p in selected)}
    q.dump(folder/'training_diagnostic_detail.json',detail)
    result['training_diagnostic_detail']=detail
    audit=json.loads((RUN/'compiled/audit.json').read_text())
    if audit.get('semantic_positive_first_new_entity_CE'):
        audit['inherited_v2_byte_identity_flag_scope']='V2 vs its pilot, not V3 metadata-augmented records'
        audit['supervised_records_byte_identical_to_pilot']=False
        audit['positive_token_trajectories_identical_to_parent']=True
        q.dump(RUN/'compiled/audit.json',audit)
    q.dump(folder/'summary.json',result);q.dump(folder/'old_causal_claim_audit.json',old_cause_audit())
    print(json.dumps({'tables':tables,'name_group_audit':names,'budgets':result['actual_budgets']},indent=2))


if __name__=='__main__':main()

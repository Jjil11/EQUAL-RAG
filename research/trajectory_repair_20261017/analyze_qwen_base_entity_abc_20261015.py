"""Fixed metrics and question-paired uncertainty, never select checkpoints."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import numpy as np
import qwen_equal_graph_20261014 as q
from qwen_base_entity_abc_data_20261015 import RUN
from score_strict_units_20261010 import score,summarize,paired
from entity_abc_contracts_20261015 import inventory,parse_closed,supported_unit,normalize


def paired_fixed(method,base,samples=20000):
    """Question bootstrap; denominator-free draws are NA for Q, not zero."""
    a_by={r['id']:r for r in method};b_by={r['id']:r for r in base}
    assert a_by.keys()==b_by.keys()
    ids=sorted(a_by);fields=['grounded_edges','gold_edges','tokens','answer_f1','repeated_units','cap_hit_proxy']
    a=np.array([[a_by[k][f] for f in fields] for k in ids],dtype=float)
    b=np.array([[b_by[k][f] for f in fields] for k in ids],dtype=float)
    assert np.array_equal(a[:,1],b[:,1])
    def calc(v):
        denominator=v[:,1].sum()
        return np.array([v[:,0].sum()/denominator if denominator else np.nan,
            v[:,2].mean(),v[:,3].mean(),np.mean(v[:,4]>0),v[:,5].mean()])
    observed=calc(a)-calc(b);rng=np.random.default_rng(20261010);draws=[]
    for _ in range(samples):
        ix=rng.integers(0,len(ids),len(ids));draws.append(calc(a[ix])-calc(b[ix]))
    draws=np.array(draws);names=['grounded_quality','mean_tokens','answer_f1','repetition_rate','true_cap_hit_rate']
    result={name:{'delta':float(observed[i]),'ci95':np.nanquantile(draws[:,i],[.025,.975]).tolist()} for i,name in enumerate(names)}
    result.update(samples=samples,zero_Q_denominator_draws=int(np.isnan(draws[:,0]).sum()),
        Q_interval_uses_only_defined_denominators=True,other_intervals_use_all_draws=True,
        method=summarize(method),baseline=summarize(base),no_noninferiority_claim=True)
    return result


def expanded_diagnostics(model):
    import hashlib
    records={r['id']:r for r in q.read(RUN/'compiled/records.jsonl')}
    inputs={r['id']:r for r in q.read(RUN/'training_source350_snapshot.jsonl')}
    ps=q.read(RUN/'diagnostic'/model/'predictions.jsonl');bytype={t:[] for t in ['A','B','C']}
    for p in ps:
        row=records[p['id']];parsed=parse_closed(p['output']);valid=False
        if parsed:
            units,tail=parsed
            if row['type'] in ['A','B']:
                n=row['past_unit_count'];entities=inventory(inputs[row['question_id']])
                key=supported_unit(units[n],entities) if len(units)>n else None
                allowed=row['state']['delta_E'] if row['type']=='A' else row['state']['E_future']
                valid=key in allowed and key not in row['state']['E_past']
            else:
                expected=parse_closed(row['positive_text'])[0]
                edge=lambda items:{(normalize(u['name']),r) for u in items for r in u['refs']}
                valid=p['ended_by_EOS'] and len(units)==row['past_unit_count'] and tail.strip() in ['', '.'] and edge(units)==edge(expected)
        root=hashlib.sha256(json.dumps(row['positive_ids'][:row['shared_tokens']]).encode()).hexdigest()
        bytype[row['type']].append({'id':p['id'],'root':root,'exact_target_success':p['conditional_prefix_success'],'valid_type_success':bool(valid)})
    result={}
    for kind,items in bytype.items():
        roots={x['root'] for x in items}
        result[kind]={'supervision_records':len(items),'unique_prefixes':len(roots),
            'exact_target_success':sum(x['exact_target_success'] for x in items),
            'valid_type_success':sum(x['valid_type_success'] for x in items),
            'unique_prefix_valid_success':sum(any(x['root']==root and x['valid_type_success'] for x in items) for root in roots),
            'definition':'A: first next unit is supported unseen entity from delta; B: first next unit is supported future entity, not repeat; C: EOS and existing name-document edges preserved.',
            'not_end_to_end_success':True}
    return result


def analyze(stage1=False):
    rows=q.read(RUN/'eval69.jsonl');byid={r['id']:r for r in rows}
    models=['base','base_block','anchor'] if stage1 else ['base','base_block','anchor','base_ce','base_pair']
    folder=RUN/('stage1_analysis' if stage1 else 'final_analysis');folder.mkdir()
    tables={};scored={};predictions={}
    for model in models:
        source=RUN/'cached_eval'/(model+'.jsonl') if model in ['base','anchor'] else RUN/'eval'/model/'predictions.jsonl'
        ps=q.read(source);assert len(ps)==69 and set(byid)=={p['id'] for p in ps}
        predictions[model]=ps
        if model not in ['base','anchor']:assert json.loads((RUN/'eval'/model/'status.json').read_text())['state']=='COMPLETE'
        baseline={p['id']:p for p in q.read(RUN/'cached_eval/base.jsonl')}
        for p in ps:assert p['prompt_token_ids']==baseline[p['id']]['prompt_token_ids']
        scores=[score(byid[p['id']],dict(p,grounded_edges=0)) for p in ps];scored[model]={s['id']:s for s in scores}
        q.write_rows(folder/(model+'_scored69.jsonl'),scores)
        group={}
        for split in ['multi45','raw24']:
            ids={r['id'] for r in rows if r['eval_split']==split};values=[s for s in scores if s['id'] in ids]
            summary=summarize(values)
            if not sum(s['gold_edges'] for s in values):summary['grounded_quality']=None;summary['utility']=None
            summary['true_cap_hit_rate']=sum(p['hit_cap_without_eos'] for p in ps if p['id'] in ids)/len(ids)
            summary['lexical_Q_scope']='Deduplicated identical alias groups, phrase-boundary visible support, strict local citation proxy; NOT entailment.'
            canonical=[]
            for p in ps:
                if p['id'] not in ids:continue
                r=byid[p['id']];entities=inventory(r)
                edges=[{'answer':g['aliases'][0],'aliases':g['aliases'],'doc_id':min(g['supporting_doc_ids']),
                    'supporting_doc_ids':g['supporting_doc_ids']} for g in entities if g['supporting_doc_ids']]
                canonical.append(score(dict(r,edges=edges),dict(p,grounded_edges=0)))
            denominator=sum(s['gold_edges'] for s in canonical)
            summary['supplemental_overlap_merged_entity_Q']=sum(s['grounded_edges'] for s in canonical)/denominator if denominator else None
            summary['overlap_merged_entity_denominator']=denominator
            group[split]=summary
        tables[model]=group
    intervals={}
    # Report Base comparisons for every arm; Pair-vs-CE is the core loss contrast.
    contrasts=[(m,'base') for m in models if m!='base']
    if not stage1:contrasts+=[('base_pair','base_ce')]
    for model,baseline in contrasts:
        for split in ['multi45','raw24']:
            ids=sorted(r['id'] for r in rows if r['eval_split']==split)
            a=[dict(scored[model][k],cap_hit_proxy=next(p['hit_cap_without_eos'] for p in predictions[model] if p['id']==k)) for k in ids]
            b=[dict(scored[baseline][k],cap_hit_proxy=next(p['hit_cap_without_eos'] for p in predictions[baseline] if p['id']==k)) for k in ids]
            result=paired_fixed(a,b,samples=20000)
            result['true_cap_note']='Bootstrap cap field replaced by telemetry-defined true cap, not mere token-count proxy.'
            intervals[model+'__minus__'+baseline+'__'+split]=result
    blocked=predictions['base_block']
    probe={'questions_with_blocking_events':sum(bool(p['blocking_events']) for p in blocked),
        'top1_blocked_events':sum(e['top1_was_blocked'] for p in blocked for e in p['blocking_events']),
        'questions_with_offline_unseen_supported_first_token_collisions':sum(bool(p['offline_unseen_supported_first_token_collisions']) for p in blocked),
        'gold_used_in_generation':False,'collision_audit_performed_only_after_generation':True}
    result={'stage':'stage1' if stage1 else 'final','tables':tables,'paired_bootstrap':intervals,
        'bootstrap_samples':20000,'bootstrap_seed':20261010,'probe_audit':probe,
        'reused_dev_not_blind_test':True,'no_multiplicity_adjustment':True,
        'semantic_exhaustion_or_answerability_certified':False,'script_sha256':q.sha(__file__)}
    if not stage1:
        result['training_diagnostics']={m:json.loads((RUN/'diagnostic'/m/'summary.json').read_text()) for m in ['base','base_ce','base_pair']}
        result['training_type_condition_diagnostics']={m:expanded_diagnostics(m) for m in ['base','base_ce','base_pair']}
        result['type_condition_metrics_supplement_original_exact_target_counts']=True
        manifests={m:json.loads((RUN/m/'manifest.json').read_text()) for m in ['base_ce','base_pair']}
        for key in ['initial_trainable_weights_sha256','data_sha256','order_sha256','updates','visits','trainable_parameter_count']:
            assert manifests['base_ce'][key]==manifests['base_pair'][key],key
        result['matched_training_budget']={k:manifests['base_ce'][k] for k in ['updates','visits','trainable_parameter_count']}
        result['same_initial_LoRA_weights']=True
        result['effective_pair_visits']={m:sum(r['pair_weight']>0 and r['pair_hinge_computed_both_arms']>0 for r in q.read(RUN/m/'visit_history.jsonl')) for m in ['base_ce','base_pair']}
    q.dump(folder/'summary.json',result)
    print(json.dumps({'stage':result['stage'],'tables':tables,'probe':probe},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--stage1',action='store_true');a=ap.parse_args();analyze(a.stage1)

"""Known-development-only strata and failed-unit semantics; never read holdout.

Lexical density is not semantic answerability. Original full-gold F1 unchanged.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import re
from pathlib import Path
from transformers import AutoTokenizer
import qwen_equal_graph_20261014 as q
import qwen_equal_curriculum_20261014 as c
import qwen_gemini_span_20261014 as m
import qwen_gemini_native_span_20261015 as n
from score_strict_units_20261010 import norm,score,summarize

RUN=q.ROOT/'runs/qwen_gemini_next_audit_20261015'
RAW=q.ROOT/'runs/qwen_decision_recovery_20261012/analysis/prospective_dev/data_full_gold_snapshot.jsonl'
REG=q.ROOT/'runs/qwen_decision_recovery_20261012/analysis/regression_dev/data_full_gold_snapshot.jsonl'
OLD=q.ROOT/'runs/qwen_decision_recovery_20261012/controls'


def lexical_groups(row):
    groups={}
    for aliases in row['all_gold_answers']:
        key=tuple(sorted({norm(a) for a in aliases if norm(a)}))
        if not key:continue
        groups.setdefault(key,list(aliases))
    edges=[]
    for key,aliases in groups.items():
        docs=[d['id'] for d in row['docs'] if any(re.search(r'(?:^| )'+re.escape(a)+r'(?: |$)',norm(d['title']+' '+d['text'])) for a in key)]
        if docs:edges.append({'answer':aliases[0],'aliases':aliases,'doc_id':min(docs),'supporting_doc_ids':sorted(docs)})
    return edges


def inventory_change(prefix,inventory):
    parsed=q.cited_units(prefix);assert parsed is not None
    old_names={norm(u['name']) for u in parsed};names={norm(u['name']) for u in inventory}
    old_units={u['key'] for u in parsed};units={(u['name'].casefold(),tuple(u['refs'])) for u in inventory}
    return {'new_names':sorted(names-old_names),'missing_observed_unit_keys':sorted(units-old_units),
        'all_observed_names_already_present':names<=old_names,
        'category':'new_name' if names-old_names else 'additional_citation_only' if units-old_units else 'inventory_complete'}


def repeat_before_new(text,edges):
    parsed=q.cited_units(text)
    if parsed is None:return None
    seen=set();repeat=False
    for i,u in enumerate(parsed):
        name=norm(u['name'])
        if name in seen:repeat=True
        elif repeat:
            known=any(name in {norm(a) for a in e['aliases']} and set(u['refs'])&set(e.get('supporting_doc_ids',[e['doc_id']])) for e in edges)
            if known:return {'new_name_after_repeat':u['name'],'citation_refs':u['refs'],
                'prefix_names':[x['name'] for x in parsed[:i]],'known_lexical_support':True}
        seen.add(name)
    return None


def main():
    assert not RUN.exists();RUN.mkdir()
    raw=q.read(RAW);train=q.read(m.SOURCE);reg=q.read(REG)
    assert len(raw)==80 and len(train)==350 and len(reg)==80
    assert q.sha(RAW)=='8662164b1aede187f7fc806a75f147f0ccac07bbd3555d9f81991fd41d125844'
    old_ids={r['id'] for r in train};old_questions={norm(r['question']) for r in train}
    assert not old_ids&{r['id'] for r in raw} and not old_questions&{norm(r['question']) for r in raw}
    assert len({r['id'] for r in raw})==len({norm(r['question']) for r in raw})==80
    rows=[]
    for r in raw:
        edges=lexical_groups(r)
        rows.append(dict(r,dedup_boundary_lexical_edges=edges,
            lexical_stratum='multiple' if len(edges)>=2 else 'sparse',
            stratum_scope='Lexical-density only, NOT verified answerability.'))
    q.write_rows(RUN/'known_dev_stratified80.jsonl',rows)
    strata={}
    for name in ['multiple','sparse']:
        subset=[r for r in rows if r['lexical_stratum']==name];counts=[len(r['dedup_boundary_lexical_edges']) for r in subset];den=sum(counts)
        strata[name]={'questions':len(subset),'deduplicated_lexical_groups':den,
            'max_one_question_groups':max(counts,default=0),
            'max_one_question_micro_Q_percentage_points':100*max(counts,default=0)/den if den else None}
    q.dump(RUN/'protocol.json',{'purpose':'Audit available reused dev, no new clean/independent benchmark claim.',
        'eligible_raw_pool':80,'stratify':'>=2 unique normalized-alias-set groups with phrase-boundary matches; otherwise0-1.',
        'no_questions_resampled_by_predictions':True,'no_sealed_inputs_read':True,'sealed_predictions':0,
        'original_full_gold_F1_preserved':True,'dedup_lexical_Q_supplemental_only':True,
        'all_strata_keep_quality_and_cost_metrics':True,'source_sha256':q.sha(RAW),'code_sha256':q.sha(__file__)})
    # Historical identical format-prefix controls, NOT the newer format-suffix condition.
    cached={};manifests={a:json.loads((OLD/a/'manifest.json').read_text()) for a in ['native_format','anchor_format']}
    for k in ['format_prefix','max_new_tokens','max_length','greedy','disable_thinking']:
        assert manifests['native_format'][k]==manifests['anchor_format'][k]
    for a,manifest in manifests.items():
        ps=q.read(OLD/a/'prospective_dev/eval_results.jsonl');ts={r['id']:r for r in q.read(OLD/a/'prospective_dev/termination_telemetry.jsonl')}
        assert len(ps)==80 and {p['id'] for p in ps}=={r['id'] for r in rows}==set(ts)
        byid={r['id']:r for r in rows};scores=[score(byid[p['id']],dict(p,grounded_edges=0)) for p in ps]
        dedup=[score(dict(byid[p['id']],edges=byid[p['id']]['dedup_boundary_lexical_edges']),dict(p,grounded_edges=0)) for p in ps]
        groups={}
        for name in ['all','multiple','sparse']:
            ids={r['id'] for r in rows if name=='all' or r['lexical_stratum']==name}
            sc=[r for r in scores if r['id'] in ids];de=[r for r in dedup if r['id'] in ids]
            summary=summarize(sc)
            summary.update(supplemental_dedup_boundary_lexical_Q=sum(r['grounded_edges'] for r in de)/sum(r['gold_edges'] for r in de) if sum(r['gold_edges'] for r in de) else None,
                true_cap_hit_rate=sum(ts[k]['hit_cap_without_eos'] for k in ids)/len(ids))
            groups[name]=summary
        cached[a]={'tables':groups,'predictions_sha256':q.sha(OLD/a/'prospective_dev/eval_results.jsonl'),
            'telemetry_sha256':q.sha(OLD/a/'prospective_dev/termination_telemetry.jsonl'),
            'condition':'Historical20261012 shared FORMAT prefix; do NOT merge with20261014 suffix prompt results.'}
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True);source={r['id']:r for r in train};failures=[];classification=[]
    failed={r['id']:r for r in q.read(n.RUN/'diagnostic/span_ce/predictions.jsonl') if not r['clean_inventory_equivalent_eos']}
    for r in q.read(n.RUN/'records.jsonl'):
        body=tok.decode(r['native_context_ids'][r['prompt_tokens']:],skip_special_tokens=True).strip()+']'
        change=inventory_change(body,r['inventory']);classification.append({'id':r['id'],'task':r['task'],**change})
        if r['id'] not in failed:continue
        src=source[r['id']];closed=failed[r['id']]['output'];positive=r['candidate_texts'][r['selected_variant']]
        scores={a:score(src,{'id':r['id'],'output':text,'tokens':0,'grounded_edges':0}) for a,text in [('stopped',closed),('positive',positive)]}
        missing=[u for u in r['inventory'] if (u['name'].casefold(),tuple(u['refs'])) in change['missing_observed_unit_keys']]
        evidence=[]
        for u in missing:
            for d in src['docs']:
                text=d['title']+' '+d['text'];normalized=norm(text)
                if norm(u['name']) in normalized:
                    evidence.append({'name':u['name'],'doc_id':d['id'],'is_requested_ref':d['id'] in u['refs'],
                        'text':text,'lexical_only_not_entailment':True})
        failures.append({'id':r['id'],'question':src['question'],'stopped':closed,'positive':positive,
            'classification':change,'missing_units':missing,'offline_score_comparison':scores,'lexical_evidence':evidence})
    q.write_rows(RUN/'failed_cases.jsonl',failures)
    examples=[]
    for r in train:
        x=repeat_before_new(r['short'],r['edges'])
        if x:examples.append(dict(id=r['id'],question=r['question'],**x))
    q.write_rows(RUN/'repeat_before_known_new_examples.jsonl',examples)
    report={'status':'PASS_CPU_AUDIT','strata':strata,'original_lexical_groups':sum(len(r['edges']) for r in raw),
        'raw80_old350_id_and_question_overlap':0,
        'reg80_overlap_old350':len(old_ids&{r['id'] for r in reg}),
        'reg80_overlap_raw80':len({r['id'] for r in reg}&{r['id'] for r in raw}),
        'reg80_not_added_to_raw_pool':'Different evidence view; rawTop5 cannot be invented from curated documents.',
        'requirements100_150_questions_and50_80_dense_met':len(raw)>=100 and strata['multiple']['questions']>=50,
        'cached_baseline_analysis':cached,'original13_native_state_classification':classification,
        'failed_case_count':len(failures),'repeat_then_known_new_counterexamples':len(examples),
        'new_model_generations':0,'new_training_runs':0,'sealed_inputs_read':0,'sealed_predictions':0,
        'code_sha256':q.sha(__file__)}
    q.dump(RUN/'cpu_audit.json',report)
    print(json.dumps({k:report[k] for k in ['strata','original_lexical_groups','reg80_overlap_old350','reg80_overlap_raw80','failed_case_count','repeat_then_known_new_counterexamples']},indent=2))


if __name__=='__main__':main()

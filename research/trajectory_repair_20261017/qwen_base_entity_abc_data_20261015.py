"""Freeze reused dev, collect Base trajectories, compile disjoint A/B/C labels."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
from pathlib import Path
import qwen_equal_graph_20261014 as q
import qwen_gemini_span_20261014 as old
from eval_qwen_recovery_controls_20261012 import FORMAT
from audit_qwen_gemini_next_20261015 import lexical_groups
from entity_abc_contracts_20261015 import inventory,parse_closed,identify,supported_unit,state_at,classify_repeat,validate_label,common_prefix,selective_positions

RUN=q.ROOT/'runs/qwen_base_entity_abc_20261015'
POOL=q.ROOT/'runs/qwen_stratified_raw_dev_20261015'
STOPS=[151643,151645]
SEED=20261015


def digest(value):return hashlib.sha256(value.encode()).hexdigest()


def prepare():
    from transformers import AutoTokenizer
    assert not RUN.exists();RUN.mkdir()
    train=q.read(old.SOURCE);pool=q.read(POOL/'known_raw_pool160.jsonl');byid={r['id']:r for r in pool}
    assert len(train)==350 and len(byid)==160
    dense=[dict(r,eval_split='multi45',prompt=FORMAT+r['prompt'],edges=lexical_groups(r)) for r in q.read(POOL/'restored_raw80.jsonl') if len(r['dedup_boundary_lexical_edges'])>=2]
    prior24=q.read(old.RUN/'validation.jsonl');assert len(prior24)==24 and all(r['view']=='raw' for r in prior24)
    ids24=[r['question_id'] for r in prior24]
    small=[dict(byid[k],eval_split='raw24',prompt=FORMAT+byid[k]['prompt'],edges=lexical_groups(byid[k])) for k in ids24]
    rows=dense+small;assert len(rows)==len({r['id'] for r in rows})==69
    assert not {r['id'] for r in rows}&{r['id'] for r in train}
    from entity_abc_contracts_20261015 import normalize
    assert not {normalize(r['question']) for r in rows}&{normalize(r['question']) for r in train}
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True)
    lengths=[len(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)) for r in rows]
    assert max(lengths)<=2048
    collection=sorted(train,key=lambda r:digest(str(SEED)+r['id']))[:128]
    collection=[dict(r,prompt=FORMAT+r['prompt']) for r in collection]
    assert max(len(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)) for r in collection)<=2048
    q.write_rows(RUN/'eval69.jsonl',rows);q.write_rows(RUN/'collection_inputs128.jsonl',collection)
    q.write_rows(RUN/'training_source350_snapshot.jsonl',train)
    # Reuse exact matched FORMAT inputs, not the old suffix-instruction Raw24.
    cache=RUN/'cached_eval';cache.mkdir()
    for arm,oldarm in [('base','native_format'),('anchor','anchor_format')]:
        new={r['id']:r for r in q.read(POOL/'baseline_comparison'/arm/'predictions.jsonl')}
        folder=q.ROOT/'runs/qwen_decision_recovery_20261012/controls'/oldarm
        manifest=json.loads((folder/'manifest.json').read_text());assert manifest['format_prefix']==FORMAT
        prior={r['id']:r for r in q.read(folder/'prospective_dev/eval_results.jsonl')}
        telemetry={r['id']:r for r in q.read(folder/'prospective_dev/termination_telemetry.jsonl')}
        result=[]
        for r in rows:
            prompt_ids=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            if r['eval_split']=='multi45':
                p=new[r['id']];assert p['prompt_token_ids']==prompt_ids
                generated=p['generated_token_ids'];origin='20261015 restored80'
            else:
                p=prior[r['id']];generated=telemetry[r['id']]['generated_token_ids'];origin='20261012 raw80 FORMAT'
            result.append(dict(p,eval_split=r['eval_split'],prompt_token_ids=prompt_ids,generated_token_ids=generated,
                ended_by_eos=bool(generated and generated[-1] in STOPS),
                hit_cap_without_eos=bool(len(generated)==256 and generated[-1] not in STOPS),reuse_source=origin))
        q.write_rows(cache/(arm+'.jsonl'),result)
    q.dump(RUN/'protocol.json',{'model':q.MODEL,'training_initialization':'Official Qwen3-14B plus fresh LoRA; NO old Anchor adapter',
        'reference':'Official Qwen3-14B, frozen distribution cache',
        'anchor_baseline_label':'Historical pref0.4 calibration adapter (user-designated naive-expansion baseline); causal attribution not established',
        'source350_sha256':q.sha(old.SOURCE),'collection_sha256':q.sha(RUN/'collection_inputs128.jsonl'),
        'eval_sha256':q.sha(RUN/'eval69.jsonl'),'eval_splits':{'multi45':45,'raw24':24},
        'lexical_group_denominators':{s:sum(len(r['edges']) for r in rows if r['eval_split']==s) for s in ['multi45','raw24']},
        'raw24_source':str(old.RUN/'validation.jsonl'),'eval_training350_overlap':0,
        'data_sources':'A from new official-Base natural-EOS trajectories; B/C from actual historical Anchor trajectories within training350 only',
        'offpolicy_B_C':True,'historical_B_C_original_prompt_retained':True,
        'entity_definition':'Connected components of overlapping normalized gold aliases; proxy, not guaranteed identity',
        'support_definition':'Phrase-boundary alias occurrence in visible title/text; NOT relation entailment',
        'C_scope':'Observed natural-EOS trajectory plus exhausted known lexical set, correct local citation coverage, no unknown prefix units; NOT semantic exhaustion',
        'A_B_no_EOS_targets':True,'BC_shared_positive_negative_token':'Skip conflicting hinge site, retain CE, report count; never optimize contradictory signs',
        'caps_per_type':{'A':48,'B':32,'C':48},'candidate_selection':'Fixed SHA order, at most2 states/type/question, no dev output selection',
        'epochs':2,'lr':5e-6,'grad_accum':8,'LoRA':{'rank':8,'alpha':16,'dropout':0,'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']},
        'kl_alpha':1.,'kl_target':.005,'kl_buffer':2,'max_KL_response_positions':16,
        'CE_weight':1.,'BC_pair_weight':1.,'A_pair_weight':0.,'margin':2.,
        'same_inputs_positive_CE_order_init_updates_and_forward_branches':True,'not_same_gradient_magnitude':True,
        'max_train_tokens':2048,'max_positive_suffix':32,'max_eval_new_tokens':256,'diagnostic_new_tokens':48,
        'decode':'greedy BF16 SDPA non-thinking','seed':SEED,'checkpoint_selection':'Final only; no dev-driven search',
        'probe':'Gold-free observed-name first-token blocking at new comma-separated item boundaries; unsafe shared-token collisions audited offline',
        'probe_is_not_a_training_gate':True,'automatic_stage2_after_structural_validation':True,
        'bootstrap':'20,000 question-paired samples; exploratory reused dev, no noninferiority claim or multiplicity correction',
        'sealed_records_used':0,'sealed_predictions':0,'new_models_other_than_Qwen3_14B':0,
        'script_sha256':q.sha(__file__)})


def compile_labels():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True)
    inputs={r['id']:r for r in q.read(RUN/'collection_inputs128.jsonl')}
    predictions=q.read(RUN/'collection/base/predictions.jsonl')
    train=q.read(RUN/'training_source350_snapshot.jsonl')
    candidates=[];skips=[]
    def add(row,body,actual,prompt_ids,positive,kind,state,target,origin,char_cut,past_count,negative_name=None):
        prompt_text=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        full=tok(prompt_text+positive,add_special_tokens=False)['input_ids']
        if kind=='C':full+=[151645]
        if full[:len(prompt_ids)]!=prompt_ids:
            skips.append({'id':row['id'],'type':kind,'reason':'prompt_boundary_retokenization'});return
        root=common_prefix(actual,full)
        if kind=='B':
            prefix_only=tok(prompt_text+body[:char_cut],add_special_tokens=False)['input_ids']
            root=min(root,common_prefix(actual,prefix_only))
        if root<len(prompt_ids) or root>=len(full) or len(full)>2048 or len(full)-root>32:
            skips.append({'id':row['id'],'type':kind,'reason':'token_span_bounds'});return
        crossing=len(prompt_ids)
        for j in range(len(prompt_ids)+1,len(actual)+1):
            if len(tok.decode(actual[len(prompt_ids):j],skip_special_tokens=True))>=char_cut:
                crossing=j;break
        rollback=crossing-root
        if kind=='C' and not 0<=rollback<=2:
            skips.append({'id':row['id'],'type':kind,'reason':'C_dynamic_rewind_exceeds2','rollback':rollback});return
        sites=[];collisions=[]
        if kind in ['B','C']:
            allowed=[full[root]];negative=[actual[root]] if root<len(actual) else []
            if set(allowed)&set(negative):collisions.append({'site':root,'token':full[root],'reason':'shared_entity_first_token_or_common_separator'})
            else:sites.append({'index':0,'allowed':allowed,'rejected':negative,'source':'actual_negative_first_action'})
        if kind=='C':
            negative=set(tok(',',add_special_tokens=False)['input_ids']+tok('],',add_special_tokens=False)['input_ids'])
            if negative_name:negative.add(tok(' '+negative_name,add_special_tokens=False)['input_ids'][0])
            negative-=set(STOPS)
            sites.append({'index':len(full)-root-1,'allowed':STOPS,'rejected':sorted(negative),
                'source':'counterfactual continuation after positive closure; not actual same negative-prefix logits'})
        record={'id':row['id']+'__'+kind+'__'+str(char_cut)+'__'+str(target or 'stop'),'question_id':row['id'],'type':kind,
            'origin':origin,'prompt_tokens':len(prompt_ids),'shared_tokens':root,
            'semantic_cut_char':char_cut,'dynamic_rollback_tokens':rollback,
            'past_unit_count':past_count,
            'actual_ids':actual,'positive_ids':full,'positive_text':positive,
            'positive_has_EOS':kind=='C','state':state,'target_entity':target,
            'negative_entity_name':negative_name,'pair_sites':sites,'pair_collisions_skipped':collisions,
            'retention_logit_positions':selective_positions(len(prompt_ids),root),
            'known_lexical_proxy_only':True,'semantic_exhaustion_certified':False}
        validate_label(record);candidates.append(record)
    for pred in predictions:
        row=inputs[pred['id']]
        if not pred['ended_by_eos']:skips.append({'id':row['id'],'type':'A','reason':'Base_not_natural_EOS'});continue
        response=[i for i in pred['generated_token_ids'] if i not in STOPS]
        body=tok.decode(response,skip_special_tokens=False);parsed=parse_closed(body)
        if parsed is None:skips.append({'id':row['id'],'type':'A','reason':'Base_unparseable'});continue
        units,tail=parsed
        if tail.strip() not in ['', '.']:continue
        entities=inventory(row);state=state_at(units,len(units),entities)
        for key in state['delta_E'][:2]:
            group=next(g for g in entities if g['key']==key)
            name,ref=group['matched_surfaces'][0]
            positive=body.rstrip().rstrip('.').rstrip()+', '+name+' ['+str(ref)+']'
            add(row,body,pred['prompt_token_ids']+pred['generated_token_ids'],pred['prompt_token_ids'],positive,'A',state,key,'new_Base_natural_EOS',len(body),len(units))
    for row in train:
        if not row.get('anchor_ended_by_eos'):continue
        prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        generated=row['generation_token_ids'];assert generated and generated[-1] in STOPS
        body=tok.decode([i for i in generated if i not in STOPS],skip_special_tokens=False)
        parsed=parse_closed(body)
        if parsed is None:continue
        units,tail=parsed
        if tail.strip() not in ['', '.']:continue
        entities=inventory(row)
        for i in range(1,len(units)):
            kind,state=classify_repeat(units,i,entities)
            if kind not in ['B','C']:continue
            prefix=body[:units[i-1]['end']]
            if kind=='B':
                target_unit=next(u for u in units[i:] if supported_unit(u,entities) in state['E_future'])
                key=supported_unit(target_unit,entities);group=next(g for g in entities if g['key']==key)
                ref=min(set(target_unit['refs'])&set(group['supporting_doc_ids']))
                positive=prefix+', '+target_unit['name']+' ['+str(ref)+']';cut=units[i]['name_start']
            else:positive=prefix+'.';cut=units[i-1]['end'];key=None
            add(row,body,prompt+generated,prompt,positive,kind,state,key,'historical_Anchor_natural_EOS_offpolicy',cut,i,units[i]['name'])
    counts={t:sum(r['type']==t for r in candidates) for t in ['A','B','C']}
    # Identical model inputs cannot get incompatible class labels.
    conflicts={}
    for r in candidates:
        key=digest(json.dumps(r['positive_ids'][:r['shared_tokens']]))
        conflicts.setdefault(key,set()).add(r['type'])
    bad={k for k,v in conflicts.items() if len(v)>1}
    eligible=[r for r in candidates if digest(json.dumps(r['positive_ids'][:r['shared_tokens']])) not in bad]
    selected=[]
    for kind,cap in [('A',48),('B',32),('C',48)]:
        used={}
        for r in sorted([x for x in eligible if x['type']==kind],key=lambda r:digest(str(SEED)+r['id'])):
            if used.get(r['question_id'],0)>=2:continue
            selected.append(r);used[r['question_id']]=used.get(r['question_id'],0)+1
            if sum(x['type']==kind for x in selected)==cap:break
    out=RUN/'compiled';out.mkdir()
    q.write_rows(out/'records.jsonl',selected);q.write_rows(out/'all_candidates.jsonl',candidates)
    audit={'candidate_counts':counts,'selected_counts':{t:sum(r['type']==t for r in selected) for t in ['A','B','C']},
        'questions':len({r['question_id'] for r in selected}),'state_label_conflicts_removed':len(bad),
        'shared_first_token_hinge_sites_skipped':sum(len(r['pair_collisions_skipped']) for r in selected),
        'A_true_new_entity_contract':True,'B_no_EOS_contract':True,'C_zero_known_remainder_contract':True,
        'B_C_offpolicy_teacher_states':True,'C_only_observed_natural_EOS_full_trajectories':True,
        'max_positive_suffix':max([len(r['positive_ids'])-r['shared_tokens'] for r in selected] or [0]),
        'C_rollback_counts':{str(k):sum(r['type']=='C' and r['dynamic_rollback_tokens']==k for r in selected) for k in [0,1,2]},
        'semantic_support_verified':False,'semantic_closure_certified':False,'skips':skips,
        'structural_gate':all(any(r['type']==t for r in selected) for t in ['A','B','C']),
        'records_sha256':q.sha(out/'records.jsonl'),'code_sha256':q.sha(__file__)}
    q.dump(out/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='skips'},indent=2),flush=True)
    assert audit['structural_gate'], 'Do not fabricate absent classes'


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--compile',action='store_true');args=ap.parse_args()
    compile_labels() if args.compile else prepare()

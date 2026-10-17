"""Native-tail calibration with exhaustive native-prefix retention.

All labels and gates are training-only lexical proxies. No runtime routing.
Historical Anchor C/B examples are never reused as native Base examples.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import qwen_equal_graph_20261014 as q
import qwen_base_entity_abc_eval_20261015 as ev
from qwen_base_entity_abc_train_20261015 import adapter_init_hash
from entity_abc_contracts_20261015 import inventory,parse_closed,identify,supported_unit,state_at,common_prefix,normalize

RUN=q.ROOT/'runs/qwen_tail_fidelity_20261015'
PREVIOUS=q.ROOT/'runs/qwen_equal_convergence_20261015'
COLLECTION=q.ROOT/'runs/qwen_base_entity_abc_20261015'
STOPS=[151643,151645]
SEED=20261015
ev.RUN=RUN


def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def positions(prompt_n,root):
    # Every generated non-edited token, including the first response action.
    return list(range(prompt_n-1,root-1))


def new_record(key,kind,prompt_n,actual,positive,text,root,past,state,entity=None):
    return {'id':key,'type':kind,'subtype':'C2' if kind=='C' else kind,
        'prompt_tokens':prompt_n,'shared_tokens':root,'repair_root_tokens':root,
        'actual_ids':actual,'positive_ids':positive,'positive_text':text,
        'positive_has_EOS':positive[-1] in STOPS,'past_unit_count':past,
        'state':state,'target_entity':entity,'pair_sites':[],
        'retention_logit_positions':positions(prompt_n,root)}


def structural_site(index,allowed,rejected,group,description):
    rejected=sorted(set(rejected)-set(allowed))
    assert rejected and not set(allowed)&set(rejected)
    return {'index':index,'allowed':sorted(set(allowed)),'rejected':rejected,
        'group':group,'source':description}


def repeated_starts(tok,names):
    return sorted({tok(' '+name.strip(),add_special_tokens=False)['input_ids'][0] for name in names})


def prepare():
    from transformers import AutoTokenizer
    assert not RUN.exists();RUN.mkdir()
    for name in ['eval69.jsonl','training_source350_snapshot.jsonl']:
        shutil.copy2(PREVIOUS/name,RUN/name)
    for name in ['cached_eval','eval/base_block']:
        shutil.copytree(PREVIOUS/name,RUN/name)
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True)
    inputs={r['id']:r for r in q.read(RUN/'training_source350_snapshot.jsonl')}
    preds=q.read(COLLECTION/'collection/base/predictions.jsonl');pmap={p['id']:p for p in preds}
    oldA=[r for r in q.read(PREVIOUS/'compiled/records.jsonl') if r['type']=='A'];assert len(oldA)==21
    A_ids={r['question_id'] for r in oldA};rows=[];native=[];candidates=[];bridges=[];audit=[]
    for before in oldA:
        key=before['question_id'];p=pmap[key];n=len(p['prompt_token_ids'])
        actual=p['prompt_token_ids']+p['generated_token_ids'];assert actual[-1] in STOPS
        body=tok.decode(p['generated_token_ids'][:-1],skip_special_tokens=False)
        parsed=parse_closed(body);assert parsed and parsed[1].strip()==''
        units=parsed[0];entities=inventory(inputs[key]);state=state_at(units,len(units),entities)
        assert state==before['state']
        order=[before['target_entity']]+sorted(set(state['delta_E'])-{before['target_entity']})
        texts=[]
        for entity in order:
            group=next(g for g in entities if g['key']==entity)
            name,ref=group['matched_surfaces'][0]
            texts.append(name.strip()+' ['+str(ref)+']')
        text=body+', '+', '.join(texts)+'.'
        prompt=tok.decode(actual[:n],skip_special_tokens=False)
        positive=tok(prompt+text,add_special_tokens=False)['input_ids']+[tok.eos_token_id]
        root=common_prefix(positive,actual);assert len(actual)-1-root in [1,2]
        assert positive[:n]==actual[:n] and tok.decode([positive[root]])=='],'
        row=new_record(before['id']+'__TAIL_COMPLETE','A',n,actual,positive,text,root,len(units),state,before['target_entity'])
        row.update(question_id=key,native_id=key,all_completion_entities=order,
            generated_prefix_retokenized=True,origin='Base_native_EOS_to_all_known_remaining_entities')
        row['pair_sites'].append(structural_site(0,[positive[root]],[actual[root]],'action','native tail closure vs native merged continuation; no noun contrast'))
        # Citation positions use tokenizer character offsets, never string-token guesses.
        encoded=tok(prompt+text,add_special_tokens=False,return_offsets_mapping=True)
        assert encoded['input_ids']==positive[:-1]
        after=parse_closed(text);assert after and len(after[0])==len(units)+len(order)
        for j,entity in enumerate(order):
            unit=after[0][len(units)+j];group=next(g for g in entities if g['key']==entity)
            citation=list(re.finditer(r'\[(\d+)\]',text[unit['name_start']:unit['end']]))[-1]
            start=len(prompt)+unit['name_start']+citation.start(1);end=start+len(citation.group(1))
            matches=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a==start and b==end]
            assert len(matches)==1
            index=matches[0];allowed=[tok(str(v),add_special_tokens=False)['input_ids'][0] for v in group['supporting_doc_ids']]
            wrong=[tok(str(v),add_special_tokens=False)['input_ids'][0] for v in range(1,6) if v not in group['supporting_doc_ids']]
            if wrong:row['pair_sites'].append(structural_site(index-root,allowed,wrong,'citation','correct supporting citation digit vs non-supporting visible document digit'))
        terminal=len(positive)-2
        assert tok.decode([positive[terminal]])=='].'
        row['C1_root_tokens']=terminal;row['C1_past_unit_count']=len(after[0])
        row['terminal_relative_positions']=[terminal-root,terminal+1-root]
        row['pair_sites'].append(structural_site(terminal-root,[positive[terminal]],[1125],'terminal','C1 exhausted known set: final closure vs continuation separator'))
        repeat=repeated_starts(tok,[u['name'] for u in after[0]])
        row['pair_sites'].append(structural_site(terminal+1-root,STOPS,[11,1125]+repeat,'terminal','C1 after closure: EOS vs contextual repeated-name/continuation starts'))
        final_state=state_at(after[0],len(after[0]),entities);assert not final_state['delta_E']
        row['C1_state']=final_state
        rows.append(row)
    for p in preds:
        if not p['ended_by_eos']:continue
        key=p['id'];actual=p['prompt_token_ids']+p['generated_token_ids'];n=len(p['prompt_token_ids'])
        body=tok.decode(p['generated_token_ids'][:-1],skip_special_tokens=False);parsed=parse_closed(body)
        if not parsed:continue
        units,tail=parsed;entities=inventory(inputs[key]);state=state_at(units,len(units),entities)
        seen=set();repeat_indices=[]
        for i,u in enumerate(units):
            if u['literal_key'] in seen:
                repeat_indices.append(i)
                st=state_at(units,i,entities)
                if st['E_future']:bridges.append({'question_id':key,'unit_index':i,'state':st})
            seen.add(u['literal_key'])
        # No old Anchor B/C is carried forward. If any native B is found,
        # compilation must implement it explicitly rather than quietly importing one.
        editable=next((r['repair_root_tokens'] for r in rows if r['question_id']==key),None)
        stop=len(actual)-1
        protect_end=editable if editable is not None else len(actual)
        # With no eligible native B, no interior position is edited or dropped.
        # Keeping even a rare repeated native token is conservative Base
        # retention, not a new anti-repetition/EOS label.
        keep=positions(n,protect_end)
        native.append({'id':key,'prompt_tokens':n,'actual_ids':actual,'positions':keep,
            'editable_tail':editable is not None,'natural_EOS':True,
            'literal_repeat_indices':repeat_indices,'known_state':state,
            'source_sha256':q.sha(COLLECTION/'collection/base/predictions.jsonl')})
        if not state['delta_E'] and not repeat_indices and tail.strip() in ['', '.'] and body.rstrip().endswith((']','].')):
            candidates.append({'prediction':p,'units':units,'state':state,'body':body,
                'unknown_prefix_entities':sum(identify(u,entities) is None for u in units)})
    # The source audit found no eligible B. Refuse silent substitution if changed.
    assert not bridges, 'New native B exists: compile and audit explicitly before training'
    selected=[]
    for sparse in [True,False]:
        pool=[r for r in candidates if (not r['state']['E_supported'])==sparse]
        pool.sort(key=lambda r:digest(str(SEED)+r['prediction']['id']))
        selected+=pool[:10]
    if len(selected)<20:
        used={r['prediction']['id'] for r in selected}
        selected+=sorted([r for r in candidates if r['prediction']['id'] not in used],key=lambda r:digest(str(SEED)+r['prediction']['id']))[:20-len(selected)]
    assert len(selected)==20
    for candidate in selected:
        p=candidate['prediction'];key=p['id'];actual=p['prompt_token_ids']+p['generated_token_ids'];n=len(p['prompt_token_ids'])
        # Preserve the actual BPE path and actual closure, including bare ] EOS.
        roots=[i for i in range(len(actual)-2,max(n-1,len(actual)-5),-1)
            if tok.decode([actual[i]]) in [']','].',']\n'] and
            tok.decode(actual[i+1:-1],skip_special_tokens=False).strip() in ['', '.']]
        assert roots
        root=roots[0]
        row=new_record(key+'__NATIVE_TAIL_C2','C',n,actual,actual.copy(),candidate['body'],root,
            len(candidate['units']),candidate['state'])
        row.update(question_id=key,native_id=key,origin='Base_native_tail_unchanged',
            unknown_prefix_entities=candidate['unknown_prefix_entities'],prefix_entities_not_relabelled_as_correct=True)
        row['retention_logit_positions']=positions(n,len(actual))
        row['pair_sites']=[structural_site(0,[actual[root]],[1125],'terminal','C2 preserve observed native closure, reject synthetic continuation'),
            structural_site(len(actual)-1-root,STOPS,[11,1125]+repeated_starts(tok,[u['name'] for u in candidate['units']]),'terminal','C2 preserve native EOS; only contextual continuation negatives')]
        rows.append(row)
    for r in rows:
        assert len(r['positive_ids'])<=2048 and r['positive_ids'][:r['shared_tokens']]==r['actual_ids'][:r['shared_tokens']]
        assert r['type']!='C' or r['positive_ids']==r['actual_ids']
        assert all(not set(s['allowed'])&set(s['rejected']) for s in r['pair_sites'])
        assert r['native_id'] in {x['id'] for x in native}
        audit.append({'id':r['id'],'type':r['type'],'native_origin_only':True,'no_internal_stop_edit':True,
            'retention_positions':len(r['retention_logit_positions']),'C1_only_after_full_known_delta':r['type']=='A',
            'PASS':True})
    dev=q.read(RUN/'eval69.jsonl')
    assert not set(inputs)&{r['id'] for r in dev}
    assert not {normalize(r['question']) for r in inputs.values()}&{normalize(r['question']) for r in dev}
    (RUN/'compiled').mkdir();q.write_rows(RUN/'compiled/records.jsonl',rows)
    q.write_rows(RUN/'compiled/native_retention.jsonl',native);q.write_rows(RUN/'compiled/checks.jsonl',audit)
    q.dump(RUN/'compiled/audit.json',{'status':'PASS','A':21,'B':0,'C1_embedded':21,'C2':20,
        'C2_candidates':len(candidates),'C2_zero_known_support':sum(not r['state']['E_supported'] for r in selected),
        'C2_unknown_prefix_entities_not_given_answer_CE':sum(r['unknown_prefix_entities'] for r in selected),
        'retention_native_trajectories':len(native),'retention_positions':sum(len(r['positions']) for r in native),
        'no_old_Anchor_examples':True,'no_forced_period_in_C2':True,'all_nonrepeat_prefix_positions_retained':True,
        'only_A_editable_original_tail_roots':len(A_ids),'no_two_token_buffer_drop':True,
        'sealed311_reads':0,'known_lexical_support_not_semantic_exhaustion':True})
    q.dump(RUN/'protocol.json',{'seed':SEED,'lr':2e-5,'grad_accum':4,'maximum_steps':60,'gate_every':5,
        'anchor_alpha':5.,'anchor_kl_target':0.,'fidelity_alpha':2.,'fidelity_margin':.5,
        'fidelity_penalty_only_argmax_changed':True,'aux_retention_weight':.5,
        'A_gate_min':12,'prefix_fidelity_min':.97,'gates_use_training_only':True,
        'independent_first_gate_stopping_both_arms':True,'do_not_train_until_800_visits':True,
        'A_first_CE_weight':1.,'span_CE_weight':1.,'C1_terminal_CE_weight':.5,
        'pair_margin':2.,'pair_weights':{'action':1.,'citation':.5,'terminal':.5},
        'records_sha256':q.sha(RUN/'compiled/records.jsonl'),
        'retention_sha256':q.sha(RUN/'compiled/native_retention.jsonl'),
        'native_source_sha256':q.sha(COLLECTION/'collection/base/predictions.jsonl'),
        'main_models_no_decoding_rules':True,'reused_development_not_blind':True,
        'C2_actual_closure_preserved_not_required_period':True,
        'C2_unknown_prefixes_only_stop_CE_not_entity_CE':True,'code_sha256':q.sha(__file__)})
    print(json.dumps(json.loads((RUN/'compiled/audit.json').read_text()),indent=2),flush=True)


def get_lp(model,ids,at):
    import torch
    import torch.nn.functional as F
    x=torch.tensor(ids,device=model.get_input_embeddings().weight.device)[None]
    logits=model(input_ids=x,use_cache=False).logits[0]
    return F.log_softmax(logits[at].float(),-1)


def cache():
    import torch
    model,tok,_=ev.load('base');model.eval();folder=RUN/'reference';folder.mkdir()
    native=q.read(RUN/'compiled/native_retention.jsonl');index={};start=time.time()
    with torch.inference_mode():
        for i,r in enumerate(native):
            lp=get_lp(model,r['actual_ids'],r['positions'])
            tops=lp.argmax(-1).cpu().tolist();torch.save(lp.cpu(),folder/(r['id']+'.pt'))
            expected=[r['actual_ids'][p+1] for p in r['positions']]
            index[r['id']]={'positions':r['positions'],'anchor_top1':tops,
                'argmax_vs_original_greedy_token_mismatches':sum(a!=b for a,b in zip(tops,expected)),
                'sha256':q.sha(folder/(r['id']+'.pt'))}
            del lp
            q.dump(folder/'status.json',{'state':'CACHING','completed':i+1,'of':len(native)})
    q.dump(folder/'index.json',index)
    q.dump(folder/'status.json',{'state':'COMPLETE','rows':len(native),'seconds':time.time()-start})


def retention_loss(lp,ref,protocol):
    import torch
    import torch.nn.functional as F
    kl=(ref.exp()*(ref-lp)).sum(-1).mean().clamp_min(0)
    anchor=ref.argmax(-1);top=lp.argmax(-1);values=lp.topk(2,-1).values
    other=torch.where(top==anchor,values[:,1],values[:,0])
    score=lp.gather(1,anchor[:,None]).squeeze(1)
    changed=top!=anchor
    fidelity=(F.relu(protocol['fidelity_margin']-(score-other))*changed).mean()
    return protocol['anchor_alpha']*kl+protocol['fidelity_alpha']*fidelity,kl,fidelity,changed.float().mean()


def supervised(model,row,protocol):
    import torch
    import torch.nn.functional as F
    n=row['shared_tokens']
    x=torch.tensor(row['positive_ids'],device=model.get_input_embeddings().weight.device)[None]
    logits=model(input_ids=x,use_cache=False).logits[0]
    lp=F.log_softmax(logits[n-1:-1].float(),-1)
    prefix=F.log_softmax(logits[row['retention_logit_positions']].float(),-1)
    targets=torch.tensor(row['positive_ids'][n:],device=lp.device)
    picked=lp.gather(1,targets[:,None]).squeeze(1)
    picked=torch.cat([picked[:-1],torch.logsumexp(lp[-1,STOPS],0)[None]])
    first=-picked[0];span=-picked.mean();terminal=span*0
    if row['type']=='A':terminal=-picked[row['terminal_relative_positions']].mean()
    pair={}
    for site in row['pair_sites']:
        pos=torch.logsumexp(lp[site['index'],site['allowed']],0)
        neg=torch.logsumexp(lp[site['index'],site['rejected']],0)
        pair.setdefault(site['group'],[]).append(F.relu(protocol['pair_margin']-(pos-neg)))
    hinge=sum(protocol['pair_weights'][group]*torch.stack(vals).mean() for group,vals in pair.items())
    ce=first+span+protocol['C1_terminal_CE_weight']*terminal
    return ce,hinge,first,span,terminal,prefix


def fidelity_probe(model,index,destination):
    import torch
    native=q.read(RUN/'compiled/native_retention.jsonl');results=[];model.eval();start=time.time()
    with torch.inference_mode():
        for r in native:
            x=torch.tensor(r['actual_ids'],device=model.get_input_embeddings().weight.device)[None]
            logits=model(input_ids=x,use_cache=False).logits[0]
            tops=logits[r['positions']].argmax(-1).cpu().tolist();wanted=index[r['id']]['anchor_top1']
            changes=[pos for pos,a,b in zip(r['positions'],tops,wanted) if a!=b]
            results.append({'id':r['id'],'positions':len(wanted),'matches':len(wanted)-len(changes),
                'changed_logit_positions':changes,'editable_A_tail':r['editable_tail']})
            del logits
    total=sum(r['positions'] for r in results);match=sum(r['matches'] for r in results)
    destination.mkdir(parents=True,exist_ok=False);q.write_rows(destination/'positions.jsonl',results)
    result={'prefix_top1_fidelity':match/total,'matched_positions':match,'total_positions':total,
        'native_trajectories':len(results),'fully_preserved_trajectories':sum(r['matches']==r['positions'] for r in results),
        'fully_preserved_trajectory_fraction':sum(r['matches']==r['positions'] for r in results)/len(results),
        'teacher_forcing_on_training_native_prefixes_not_unseen_guarantee':True,'seconds':time.time()-start}
    q.dump(destination/'summary.json',result);return result


def unit_prefix(text,count):
    for match in re.finditer(r'\[\d+\](?:\s*\[\d+\])*',text):
        parsed=parse_closed(text[:match.end()])
        if parsed and len(parsed[0])==count:return parsed[0]
    return None


def diagnose(model,tok,arm,destination):
    import torch
    model.eval();model.config.use_cache=True;rows=q.read(RUN/'compiled/records.jsonl')
    inputs={r['id']:r for r in q.read(RUN/'training_source350_snapshot.jsonl')}
    output=[];c1=[];generated={};start=time.time()
    with torch.inference_mode():
        for row in rows:
            root=row['repair_root_tokens'];context=row['positive_ids'][:root]
            key=tuple(context)
            if key not in generated:
                x=torch.tensor(context,device=model.get_input_embeddings().weight.device)[None]
                ids=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=96,do_sample=False,pad_token_id=tok.eos_token_id)[0,root:].tolist()
                generated[key]=ids
            ids=generated[key];body=tok.decode(context[row['prompt_tokens']:]+ids,skip_special_tokens=True)
            entities=inventory(inputs[row['question_id']]);k=row['past_unit_count']
            expected=parse_closed(row['positive_text'])[0]
            edge=lambda us:{(normalize(u['name']),ref) for u in us for ref in u['refs']}
            valid=False;exact=False;complete=False
            if row['type']=='A':
                units=unit_prefix(body,k+1)
                if units:
                    key=supported_unit(units[k],entities)
                    preserve=edge(units[:k])==edge(expected[:k])
                    valid=preserve and key in row['state']['delta_E'] and key not in row['state']['E_past']
                    exact=preserve and key==row['target_entity']
                parsed=parse_closed(body)
                if parsed:
                    units,tail=parsed
                    full={supported_unit(u,entities) for u in units}-{None}
                    complete=bool(ids and ids[-1] in STOPS and set(row['state']['E_supported'])<=full and tail.strip() in ['', '.'] and len(units)==len(expected))
            else:
                parsed=parse_closed(body)
                if parsed:
                    units,tail=parsed
                    valid=bool(ids and ids[-1] in STOPS and len(units)==k and tail.strip() in ['', '.'] and edge(units)==edge(expected))
                    exact=valid;complete=valid
            output.append({'id':row['id'],'type':row['type'],'output':body,'generated_ids':ids,
                'conditional_prefix_success':bool(exact),'valid_type_success':bool(valid),
                'complete_chain_success':bool(complete),'ended_by_EOS':bool(ids and ids[-1] in STOPS)})
            if row['type']=='A':
                root=row['C1_root_tokens'];context=row['positive_ids'][:root]
                x=torch.tensor(context,device=model.get_input_embeddings().weight.device)[None]
                tail_ids=model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)[0,root:].tolist()
                body=tok.decode(context[row['prompt_tokens']:]+tail_ids,skip_special_tokens=True);parsed=parse_closed(body)
                success=bool(parsed and tail_ids and tail_ids[-1] in STOPS and parsed[1].strip() in ['', '.'] and edge(parsed[0])==edge(expected) and len(parsed[0])==len(expected))
                c1.append({'id':row['id'],'output':body,'generated_ids':tail_ids,'success':success})
    destination.mkdir(parents=True,exist_ok=False);q.write_rows(destination/'predictions.jsonl',output)
    q.write_rows(destination/'C1_predictions.jsonl',c1)
    result={'model':arm,'by_type':{},'C1':{'n':len(c1),'success':sum(r['success'] for r in c1)},
        'training_only':True,'A_valid_next_entity_not_exact_target':True,'seconds':time.time()-start}
    for kind in ['A','B','C']:
        rs=[r for r in output if r['type']==kind]
        result['by_type'][kind]={'n':len(rs),'valid_type_success':sum(r['valid_type_success'] for r in rs),
            'conditional_prefix_success':sum(r['conditional_prefix_success'] for r in rs),
            'complete_chain_success':sum(r['complete_chain_success'] for r in rs)}
    q.dump(destination/'summary.json',result);q.dump(destination/'status.json',{'state':'COMPLETE'})
    model.config.use_cache=False;return result


def train(arm):
    import torch
    protocol=json.loads((RUN/'protocol.json').read_text());rows=q.read(RUN/'compiled/records.jsonl')
    native=q.read(RUN/'compiled/native_retention.jsonl');index=json.loads((RUN/'reference/index.json').read_text())
    model,tok,_=ev.load(arm,train=True);folder=RUN/arm;folder.mkdir()
    init=adapter_init_hash(model);optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=protocol['lr'])
    sup_order=[];aux_order=[];epoch=0;maximum=protocol['maximum_steps'];accum=protocol['grad_accum']
    while len(sup_order)<maximum*accum:
        sup_order+=sorted(rows,key=lambda r:digest(str(SEED+epoch)+r['id']));epoch+=1
    epoch=0
    while len(aux_order)<maximum*accum:
        aux_order+=sorted(native,key=lambda r:digest(str(SEED+epoch)+'retention'+r['id']));epoch+=1
    start=time.time();history=[];selected=None;gates=[];optimizer.zero_grad(set_to_none=True)
    with (folder/'visit_history.jsonl').open('x',encoding='utf-8') as stream:
        for step in range(1,maximum+1):
            model.train()
            for j in range(accum):
                r=sup_order[(step-1)*accum+j]
                ce,hinge,first,span,terminal,lp=supervised(model,r,protocol)
                # By causality all retained prefix IDs are exactly native; no
                # teacher-forced edited suffix position is KL-protected.
                own=next(x for x in native if x['id']==r['native_id'])
                assert own['positions']==r['retention_logit_positions']
                assert all(r['positive_ids'][:pos+1]==own['actual_ids'][:pos+1] for pos in own['positions'])
                ref=torch.load(RUN/'reference'/(own['id']+'.pt'),map_location=lp.device,weights_only=True)
                penalty,kl,fid,flip=retention_loss(lp,ref,protocol)
                weight=1. if arm=='base_pair' else 0.
                loss=ce+weight*hinge+penalty
                assert torch.isfinite(loss)
                (loss/accum).backward()
                item={'step':step,'id':r['id'],'type':r['type'],'ce':float(ce.detach()),'first_ce':float(first.detach()),
                    'span_ce':float(span.detach()),'C1_terminal_ce':float(terminal.detach()),'pair_hinge':float(hinge.detach()),
                    'pair_weight':weight,'selective_kl':float(kl.detach()),'fidelity_penalty':float(fid.detach()),
                    'supervised_prefix_flip_fraction':float(flip.detach())}
                del ce,hinge,first,span,terminal,lp,ref,penalty,kl,fid,flip,loss
                extra=aux_order[(step-1)*accum+j]
                lp=get_lp(model,extra['actual_ids'],extra['positions'])
                ref=torch.load(RUN/'reference'/(extra['id']+'.pt'),map_location=lp.device,weights_only=True)
                penalty,kl,fid,flip=retention_loss(lp,ref,protocol)
                (protocol['aux_retention_weight']*penalty/accum).backward()
                item.update(aux_native_id=extra['id'],aux_kl=float(kl.detach()),aux_flip_fraction=float(flip.detach()))
                stream.write(json.dumps(item)+'\n');stream.flush()
                del lp,ref,penalty,kl,fid,flip
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert torch.isfinite(torch.tensor(norm))
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            entry=dict(item,update=step,grad_norm=norm,seconds=time.time()-start);history.append(entry)
            q.dump(folder/'status.json',dict(entry,state='TRAINING'));print(json.dumps(entry),flush=True)
            if step%protocol['gate_every']==0:
                folder_gate=folder/('gate_'+str(step))
                fidelity=fidelity_probe(model,index,folder_gate/'fidelity')
                diagnostic=diagnose(model,tok,arm,folder_gate/'diagnostic')
                passed=diagnostic['by_type']['A']['valid_type_success']>=protocol['A_gate_min'] and fidelity['prefix_top1_fidelity']>=protocol['prefix_fidelity_min']
                gate={'step':step,'passed':passed,'diagnostic':diagnostic,'fidelity':fidelity}
                q.dump(folder_gate/'gate.json',gate);q.dump(folder/'gate_latest.json',gate);gates.append(gate)
                print(json.dumps({'gate':gate}),flush=True)
                model.save_pretrained(folder/('checkpoint_'+str(step)))
                if passed:selected=step;break
    completed=len(history);model.save_pretrained(folder/'adapter')
    q.dump(folder/'manifest.json',{'arm':arm,'starting_model':q.MODEL,'reference':'Official Base Qwen3-14B',
        'old_Anchor_loaded':False,'initial_trainable_weights_sha256':init,
        'final_trainable_weights_sha256':adapter_init_hash(model),'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'data_sha256':protocol['records_sha256'],'retention_sha256':protocol['retention_sha256'],
        'order_sha256':digest(json.dumps([r['id'] for r in sup_order[:completed*accum]])),
        'aux_order_sha256':digest(json.dumps([r['id'] for r in aux_order[:completed*accum]])),
        'updates':completed,'supervised_visits':completed*accum,'aux_retention_visits':completed*accum,
        'gate_selected_step':selected,'history':history,'script_sha256':q.sha(__file__),
        'adapter_sha256':q.sha(folder/'adapter/adapter_model.safetensors'),'gate_passed':selected is not None})
    final_gate=folder/('gate_'+str(completed));assert final_gate.exists()
    shutil.copytree(final_gate/'diagnostic',RUN/'diagnostic'/arm)
    shutil.copytree(final_gate/'fidelity',RUN/'fidelity'/arm)
    q.dump(folder/'status.json',{'state':'COMPLETE' if selected else 'GATE_FAILED','updates':completed})


def child(args,label,devices='1,3'):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    log=(q.ROOT/'logs'/('qwen_tail_fidelity_'+label+'_20261015.log')).open('x')
    process=subprocess.Popen([sys.executable,__file__,*args],cwd=q.ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
    return process,log


def pipeline():
    prepare();q.dump(RUN/'pipeline_status.json',{'state':'CACHE_REFERENCE'})
    p,log=child(['--cache'],'cache');assert p.wait()==0;log.close()
    p,log=child(['--diagnose','base'],'base_diagnostic');assert p.wait()==0;log.close()
    q.dump(RUN/'pipeline_status.json',{'state':'INDEPENDENT_TRAINING_ONLY_SWEET_SPOT_GATES'})
    workers=[child(['--train',arm],arm,devices) for arm,devices in [('base_ce','0,2'),('base_pair','1,3')]]
    codes=[]
    for p,log in workers:codes.append(p.wait());log.close()
    assert codes==[0,0],codes
    gates={arm:json.loads((RUN/arm/'gate_latest.json').read_text()) for arm in ['base_ce','base_pair']}
    if not all(g['passed'] for g in gates.values()):
        q.dump(RUN/'pipeline_status.json',{'state':'GATE_FAILED_NO_NEW_DEV_GENERATION','gates':gates});return
    q.dump(RUN/'pipeline_status.json',{'state':'GATED_FREE_GENERATION','steps':{k:g['step'] for k,g in gates.items()}})
    workers=[child(['--generate',arm],arm+'_eval',devices) for arm,devices in [('base_ce','0,2'),('base_pair','1,3')]]
    codes=[]
    for p,log in workers:codes.append(p.wait());log.close()
    assert codes==[0,0],codes
    q.dump(RUN/'pipeline_status.json',{'state':'GENERATIONS_COMPLETE_READY_FOR_ANALYSIS',
        'sealed_predictions':0,'main_models_use_rules':False,'new_dev_predictions':138})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pipeline',action='store_true');ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--cache',action='store_true');ap.add_argument('--train',choices=['base_ce','base_pair'])
    ap.add_argument('--diagnose',choices=['base']);ap.add_argument('--generate',choices=['base_ce','base_pair']);args=ap.parse_args()
    if args.pipeline:pipeline()
    elif args.prepare:prepare()
    elif args.cache:cache()
    elif args.train:train(args.train)
    elif args.diagnose:
        model,tok,_=ev.load('base');diagnose(model,tok,'base',RUN/'diagnostic/base')
    elif args.generate:
        assert all(json.loads((RUN/arm/'gate_latest.json').read_text())['passed'] for arm in ['base_ce','base_pair'])
        ev.generate(args.generate)

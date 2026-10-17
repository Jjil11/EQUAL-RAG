"""Native Mistral-7B state compilation and fixed 60-step transfer experiment."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
from collections import Counter
import gc
import re
from supp_runtime import *
from entity_abc_contracts_20261015 import inventory, parse_closed, identify, state_at, common_prefix

ROOT2 = SUITE/'package3'

def hkey(r):
    return hashlib.sha256((str(SEED)+r['id']).encode()).hexdigest()

def encode(tok,text):
    return tok(text, add_special_tokens=False, return_offsets_mapping=True)

def compile_records(tok, train_rows, predictions, eos):
    inputs={r['id']:r for r in train_rows}
    A,C2,native,candidate_log=[],[],[],[]
    for p in predictions:
        if not p['ended_by_eos']:
            candidate_log.append({'id':p['id'],'eligible':False,'reason':'cap_or_non_EOS'})
            continue
        actual=p['prompt_token_ids']+p['generated_token_ids']
        n=len(p['prompt_token_ids'])
        prompt=tok.decode(actual[:n],skip_special_tokens=False)
        complete=tok.decode(actual[:-1],skip_special_tokens=False)
        assert complete.startswith(prompt), 'Native full decode must retain prompt text'
        body=complete[len(prompt):]  # Preserve SentencePiece leading whitespace across boundary.
        base={'id':p['id'],'prompt_tokens':n,'actual_ids':actual,'positions':list(range(n-1,len(actual)-1))}
        native.append(base)
        parsed=parse_closed(body)
        if not parsed or parsed[1].strip() not in ['', '.']:
            candidate_log.append({'id':p['id'],'eligible':False,'reason':'not_a_closed_cited_list'})
            continue
        units=parsed[0]
        if len({u['literal_key'] for u in units}) != len(units):
            candidate_log.append({'id':p['id'],'eligible':False,'reason':'literal_repetition_in_native_prefix'})
            continue
        entities=inventory(inputs[p['id']])
        state=state_at(units,len(units),entities)
        if state['delta_E']:
            order_entities=sorted(state['delta_E'])
            extra=[]
            for key in order_entities:
                g=next(g for g in entities if g['key']==key)
                name,cite=g['matched_surfaces'][0]
                extra.append(name.strip()+' ['+str(cite)+']')
            text=body.rstrip().rstrip('.').rstrip()+', '+', '.join(extra)+'.'
            enc=encode(tok,prompt+text)
            positive=enc['input_ids']+[tok.eos_token_id]
            root=common_prefix(actual,positive)
            if root<n or len(actual)-1-root not in [1,2,3,4] or len(positive)>2048:
                candidate_log.append({'id':p['id'],'eligible':False,'reason':'nonlocal_retokenization_or_length'})
                continue
            after=parse_closed(text)
            if not after or len(after[0])!=len(units)+len(extra):
                candidate_log.append({'id':p['id'],'eligible':False,'reason':'repair_parser_mismatch'})
                continue
            first=after[0][len(units)]
            begin=len(prompt)+first['name_start']
            end=begin+len(first['name'])
            semantic=[i-root for i,(a,b) in enumerate(enc['offset_mapping']) if b>begin and a<end and i>=root]
            close=len(prompt)+after[0][-1]['end']-1
            terminal=[i-root for i,(a,b) in enumerate(enc['offset_mapping']) if b>close and i>=root]
            terminal.append(len(positive)-1-root)
            assert semantic and terminal and positive[root]!=actual[root]
            sites=[{'index':0,'allowed':[positive[root]],'rejected':[actual[root]],'group':'action'}]
            for j,key in enumerate(order_entities):
                unit=after[0][len(units)+j]
                cite_match=list(re.finditer(r'\[(\d+)\]',text[unit['name_start']:unit['end']]))[-1]
                digit_start=len(prompt)+unit['name_start']+cite_match.start(1)
                digit_end=digit_start+len(cite_match.group(1))
                pos=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>digit_start and a<digit_end]
                if len(pos)!=1:
                    continue
                at=pos[0]
                g=next(g for g in entities if g['key']==key)
                allowed,rejected=[],[]
                for doc in range(1,6):
                    candidate=(prompt+text)[:digit_start]+str(doc)+(prompt+text)[digit_end:]
                    ids=encode(tok,candidate)['input_ids']
                    if ids[:at]==positive[:at] and len(ids)>at:
                        (allowed if doc in g['supporting_doc_ids'] else rejected).append(ids[at])
                allowed=sorted(set(allowed));rejected=sorted(set(rejected)-set(allowed))
                if allowed and rejected:
                    sites.append({'index':at-root,'allowed':allowed,'rejected':rejected,'group':'citation'})
            first_terminal=root+terminal[0]
            closure_allowed={positive[first_terminal]}
            # Derive the native bare-bracket alternative in its REAL context.
            bare=encode(tok,prompt+text[:after[0][-1]['end']])['input_ids']
            if len(bare)>first_terminal and bare[:first_terminal]==positive[:first_terminal]:
                closure_allowed.add(bare[first_terminal])
            r={'id':p['id']+'__A','question_id':p['id'],'native_id':p['id'],'type':'A',
               'prompt_tokens':n,'actual_ids':actual,'positive_ids':positive,'positive_text':text,
               'shared_tokens':root,'repair_root_tokens':root,'retention_logit_positions':list(range(n-1,root-1)),
               'semantic_positive_positions':semantic,'terminal_relative_positions':terminal,
               'C1_closure_allowed':sorted(closure_allowed),'pair_sites':sites,'state':state,
               'all_completion_entities':order_entities,'past_unit_count':len(units),
               'rollback_tokens':len(actual)-1-root}
            assert not set(order_entities)&set(state['E_past'])
            A.append(r)
        else:
            alternative=encode(tok,prompt+body.rstrip().rstrip('.').rstrip()+', ')['input_ids']
            root=common_prefix(actual,alternative)
            if root<n or len(actual)-root>5 or root==len(actual):
                candidate_log.append({'id':p['id'],'eligible':False,'reason':'C2_not_a_local_native_tail'})
                continue
            C2.append({'id':p['id']+'__C2','question_id':p['id'],'native_id':p['id'],'type':'C',
                'prompt_tokens':n,'actual_ids':actual,'positive_ids':actual.copy(),'positive_text':body,
                'shared_tokens':root,'repair_root_tokens':root,'retention_logit_positions':base['positions'],
                'pair_sites':[],'state':state,'past_unit_count':len(units)})
        candidate_log.append({'id':p['id'],'eligible':True,'delta':len(state['delta_E'])})
    A=sorted(A,key=hkey);C2=sorted(C2,key=hkey)
    counts={'eligible_A':len(A),'eligible_C2':len(C2),'native_EOS_trajectories':len(native)}
    selectedA=A[:21]
    # Match the original zero-support/nonzero-support C2 strata where available.
    selectedC=[]
    for zero in [True,False]:
        pool=[r for r in C2 if (not r['state']['E_supported'])==zero]
        selectedC+=pool[:10]
    used={r['id'] for r in selectedC}
    selectedC+=[r for r in C2 if r['id'] not in used][:20-len(selectedC)]
    records=selectedA+selectedC
    assert selectedA and selectedC, 'No valid transfer training labels; do not fabricate labels'
    lookup={r['id']:r for r in native}
    required={r['native_id'] for r in records}
    selected_native=[lookup[k] for k in sorted(required)]
    selected_native+=[r for r in sorted(native,key=hkey) if r['id'] not in required][:120-len(selected_native)]
    edit={r['native_id']:r['shared_tokens'] for r in selectedA}
    for r in selected_native:
        if r['id'] in edit:
            r['positions']=list(range(r['prompt_tokens']-1,edit[r['id']]-1))
    for r in records:
        own=next(n for n in selected_native if n['id']==r['native_id'])
        assert own['positions']==r['retention_logit_positions']
        assert r['positive_ids'][:r['shared_tokens']]==r['actual_ids'][:r['shared_tokens']]
        assert all(r['positive_ids'][:k+1]==own['actual_ids'][:k+1] for k in own['positions'])
        assert all(not set(s['allowed'])&set(s['rejected']) for s in r['pair_sites'])
    counts.update(A=len(selectedA),C1_embedded=len(selectedA),C2=len(selectedC),B=0,
                  total_records=len(records),retention_trajectories=len(selected_native),
                  retention_positions=sum(len(r['positions']) for r in selected_native),
                  selection='fixed seed-hash caps 21 A,20 C2,120 retention; no test-score gate',
                  known_lexical_inventory_not_exhaustive_semantic_evidence=True)
    rows(ROOT2/'compiled/records.jsonl',records)
    rows(ROOT2/'compiled/native_retention.jsonl',selected_native)
    rows(ROOT2/'compiled/candidate_audit.jsonl',candidate_log)
    dump(ROOT2/'compiled/counts.json',counts)
    return records,selected_native

def prepare():
    import torch
    ROOT2.mkdir(exist_ok=False)
    model,tok,mapping=load('mistral')
    model_files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(Path(MISTRAL).iterdir()) if p.is_file()}
    dump(ROOT2/'model_identity.json',{'model':'mistralai/Mistral-7B-Instruct-v0.3','path':MISTRAL,
        'revision':Path(MISTRAL).name,'files':model_files,'placement':mapping,'tokenizer':tok.__class__.__name__})
    train_rows=read(SOURCE/'training_source350_snapshot.jsonl')
    assert len(train_rows)==350
    inference(model,tok,train_rows,ROOT2/'native_train350/predictions.jsonl')
    rr,native=compile_records(tok,train_rows,read(ROOT2/'native_train350/predictions.jsonl'),stops(model,tok))
    dump(ROOT2/'training_freeze.json',{'records_sha256':sha(ROOT2/'compiled/records.jsonl'),
        'native_retention_sha256':sha(ROOT2/'compiled/native_retention.jsonl'),
        'counts':js(ROOT2/'compiled/counts.json'),'script_sha256':sha(__file__),
        'test_outputs_inspected_before_label_freeze':False,'updates':60,'seed':SEED})
    model.eval()
    refdir=safe(ROOT2/'reference');refdir.mkdir(exist_ok=False)
    index={}
    with torch.inference_mode():
        for i,r in enumerate(native):
            x=torch.tensor(r['actual_ids'],device=model.get_input_embeddings().weight.device)[None]
            lp=torch.log_softmax(model(input_ids=x,use_cache=False).logits[0,r['positions']].float(),-1)
            destination=refdir/(r['id']+'.pt')
            torch.save(lp.cpu(),destination)
            r['base_top1']=lp.argmax(-1).cpu().tolist()
            index[r['id']]={'positions':r['positions'],'base_top1':r['base_top1'],'sha256':sha(destination)}
            if (i+1)%10==0:print(json.dumps({'phase':'cache_Mistral_retention','completed':i+1}),flush=True)
    rows(ROOT2/'compiled/native_retention_with_top1.jsonl',native)
    dump(refdir/'index.json',index)
    diagnose(model,rr,ROOT2/'base/diagnostic')
    full_fidelity(model,native,ROOT2/'base/diagnostic')
    # The base evaluation is not used to tune labels, hyperparameters or checkpoints.
    inference(model,tok,eval_rows(),ROOT2/'base/eval/predictions380.jsonl')
    dump(ROOT2/'prepare_COMPLETE.json',{'state':'COMPLETE','counts':js(ROOT2/'compiled/counts.json')})

def guard(lp,ref):
    import torch
    import torch.nn.functional as F
    if not lp.numel():return lp.sum()*0
    kl=(ref.exp()*(ref-lp)).sum(-1).mean().clamp_min(0)
    anchor=ref.argmax(-1);top=lp.argmax(-1)
    values=lp.topk(2,-1).values
    other=torch.where(top==anchor,values[:,1],values[:,0])
    changed=top!=anchor
    margin=lp.gather(1,anchor[:,None]).squeeze(1)-other
    return 10*kl+2*(F.relu(.5-margin)*changed).sum()/changed.sum().clamp_min(1)

def repair_loss(model,r,beta,eos):
    import torch
    import torch.nn.functional as F
    n=r['shared_tokens']
    x=torch.tensor(r['positive_ids'],device=model.get_input_embeddings().weight.device)[None]
    logits=model(input_ids=x,use_cache=False).logits[0]
    lp=F.log_softmax(logits[n-1:-1].float(),-1)
    targets=x[0,n:].to(lp.device)
    picked=lp.gather(1,targets[:,None]).squeeze(1)
    picked=torch.cat([picked[:-1],torch.logsumexp(lp[-1,eos],0)[None]])
    terminal=semantic=picked.sum()*0
    if r['type']=='A':
        at=r['terminal_relative_positions'][0]
        close=torch.logsumexp(lp[at,r['C1_closure_allowed']],0)
        picked=torch.cat([picked[:at],close[None],picked[at+1:]])
        terminal=-picked[r['terminal_relative_positions']].mean()
        semantic=-picked[r['semantic_positive_positions']].mean()
    ce=-picked[0]-picked.mean()+2*terminal+semantic
    groups={}
    for s in r['pair_sites']:
        pos=torch.logsumexp(lp[s['index'],s['allowed']],0)
        neg=torch.logsumexp(lp[s['index'],s['rejected']],0)
        groups.setdefault(s['group'],[]).append(F.relu(2-(pos-neg)))
    hinge=ce*0
    for group,values in groups.items():
        hinge+=beta*(1 if group=='action' else .5)*torch.stack(values).mean()
    prefix=F.log_softmax(logits[r['retention_logit_positions']].float(),-1)
    return ce+hinge,prefix

def full_fidelity(model,native,target):
    import torch
    out=[]
    model.eval()
    with torch.inference_mode():
        for r in native:
            x=torch.tensor(r['actual_ids'],device=model.get_input_embeddings().weight.device)[None]
            top=model(input_ids=x,use_cache=False).logits[0,r['positions']].argmax(-1).tolist()
            changed=sum(a!=b for a,b in zip(top,r['base_top1']))
            out.append({'id':r['id'],'positions':len(top),'changed':changed,'exact':changed==0})
    rows(target/'retention_full_contexts.jsonl',out)
    result={'exact':sum(v['exact'] for v in out),'trajectories':len(out),
            'changed':sum(v['changed'] for v in out),'positions':sum(v['positions'] for v in out)}
    dump(target/'retention_summary.json',result)
    return result

def diagnose(model,rr,target):
    import torch
    model.eval();out=[]
    with torch.inference_mode():
        for r in rr:
            x=torch.tensor(r['positive_ids'],device=model.get_input_embeddings().weight.device)[None]
            top=model(input_ids=x,use_cache=False).logits[0].argmax(-1).tolist()
            root=r['shared_tokens']
            action=top[root-1]==r['positive_ids'][root]
            record={'id':r['id'],'type':r['type'],'first_action':action}
            if r['type']=='A':
                ti=r['terminal_relative_positions']
                record['C1_success']=(top[root+ti[0]-1] in r['C1_closure_allowed'] and top[-2] in [2])
                record['first_entity_token_success']=all(top[root+i-1]==r['positive_ids'][root+i] for i in r['semantic_positive_positions'])
            else:
                record['C2_success']=all(top[i-1]==r['positive_ids'][i] for i in range(root,len(r['positive_ids'])))
            out.append(record)
    rows(target/'training_diagnostics.jsonl',out)
    result={'A_first_action':sum(v['first_action'] for v in out if v['type']=='A'),
            'A_total':sum(v['type']=='A' for v in out),'C1':sum(v.get('C1_success',False) for v in out),
            'A_first_entity_exact':sum(v.get('first_entity_token_success',False) for v in out),
            'C2':sum(v.get('C2_success',False) for v in out),'C2_total':sum(v['type']=='C' for v in out)}
    dump(target/'training_diagnostics_summary.json',result)
    return result

def train(arm):
    import torch
    from qwen_base_entity_abc_train_20261015 import adapter_init_hash
    folder=ROOT2/arm;folder.mkdir(exist_ok=False)
    rr=read(ROOT2/'compiled/records.jsonl')
    native=read(ROOT2/'compiled/native_retention_with_top1.jsonl')
    visits=order(rr);extras=order(native,'retention')
    model,tok,mapping=load('mistral',train=True)
    init=adapter_init_hash(model)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=2e-5)
    byid={r['id']:r for r in native}
    beta=.25 if arm=='equal_a025' else 0.
    started=time.time();optimizer.zero_grad(set_to_none=True)
    with safe(folder/'visits.jsonl').open('x',encoding='utf-8') as f:
        for step in range(60):
            model.train()
            for j,r in enumerate(visits[step*4:step*4+4]):
                if arm=='matched_sft':
                    logp,count=sequence_logp(model,r['positive_ids'],r['prompt_tokens'])
                    loss=-logp/count
                else:
                    loss,lp=repair_loss(model,r,beta,stops(model,tok))
                    ref=torch.load(ROOT2/'reference'/(r['native_id']+'.pt'),map_location=lp.device,weights_only=True)
                    loss=loss+guard(lp,ref)
                assert torch.isfinite(loss)
                value=float(loss.detach());(loss/4).backward()
                del loss
                if arm!='matched_sft':
                    del lp,ref
                    extra=extras[step*4+j]
                    x=torch.tensor(extra['actual_ids'],device=model.get_input_embeddings().weight.device)[None]
                    lp=torch.log_softmax(model(input_ids=x,use_cache=False).logits[0,extra['positions']].float(),-1)
                    ref=torch.load(ROOT2/'reference'/(extra['id']+'.pt'),map_location=lp.device,weights_only=True)
                    (.5*guard(lp,ref)/4).backward()
                    del x,lp,ref
                f.write(json.dumps({'step':step+1,'id':r['id'],'type':r['type'],'loss':value})+'\n');f.flush()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            assert torch.isfinite(torch.tensor(norm))
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            dump(folder/'training_status.json',{'step':step+1,'seconds':time.time()-started},replace=True)
            print(json.dumps({'arm':arm,'step':step+1,'seconds':time.time()-started}),flush=True)
    model.save_pretrained(safe(folder/'adapter'))
    dump(folder/'manifest.json',{'model':MISTRAL,'arm':arm,'beta':beta,'updates':60,'supervised_visits':240,
         'aux_retention_visits':0 if arm=='matched_sft' else 240,'seed':SEED,'initial_adapter_sha256':init,
         'adapter_sha256':sha(folder/'adapter/adapter_model.safetensors'),'order_sha256':order_sha(visits),
         'records_sha256':sha(ROOT2/'compiled/records.jsonl'),'unmerged':True,'training_seconds':time.time()-started,
         'script_sha256':sha(__file__)})
    diagnose(model,rr,folder/'diagnostic')
    full_fidelity(model,native,folder/'diagnostic')
    # Same model after training, with no merging or decoding modification.
    inference(model,tok,eval_rows(),folder/'eval/predictions380.jsonl')
    dump(folder/'COMPLETE.json',{'state':'COMPLETE','predictions_sha256':sha(folder/'eval/predictions380.jsonl')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['prepare','matched_sft','equal_ce','equal_a025'])
    a=ap.parse_args()
    prepare() if a.task=='prepare' else train(a.task)

"""Bounded Gemini-inspired repair-span mechanism pilot, NOT final efficacy.

Uses naturally ended real teacher trajectories, preserves all observed units,
and keeps late distinct units instead of forcing stop at the first repeat.
Matched span CE vs normalized full-span margin plus terminal competition.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import qwen_equal_graph_20261014 as q
from audit_qwen_gemini_20261014 import SOURCE
from compile_qwen_coherent_span_20261012 import common_prefix

RUN=q.ROOT/'runs/qwen_gemini_span_20261014'
ARMS=['span_ce','span_pair']
STOPS=[151643,151645]


def buffer_positions(prompt_tokens,shared_tokens,buffer=2):
    # Absolute LOGIT positions predict the following token. Never interpret
    # cached-response mask coordinates as original-generation coordinates.
    return list(range(prompt_tokens-1,max(prompt_tokens-1,shared_tokens-buffer-1)))


def terminal_competition(lp,allowed=STOPS,margin=2.):
    import torch
    import torch.nn.functional as F
    mask=lp.detach().clone();mask[allowed]=-float('inf')
    rejected=int(mask.argmax())
    assert rejected not in allowed
    positive=torch.logsumexp(lp[allowed],0)
    return F.relu(margin-(positive-lp[rejected])),rejected


def compile_data():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True);groups={'stop':[],'continue':[]};skips=[]
    for row in q.read(SOURCE):
        d=row.get('terminal_decision',{})
        if not row.get('has_trajectory') or d.get('origin')!='recorded_onpolicy_prefix':continue
        if not row.get('anchor_ended_by_eos',False):skips.append({'id':row['id'],'reason':'original_cap_no_EOS_prior'});continue
        edit=q.compact_equivalent(row['short'])
        if edit is None or not edit['removed_duplicates']:skips.append({'id':row['id'],'reason':'not_parseable_repetition'});continue
        original=q.cited_units(row['short']);inventory={u['key'] for u in original}
        body=edit['chosen'].rstrip().rstrip('.').rstrip()
        variants=[body+s for s in ['', '.', '\n']]
        assert all({u['key'] for u in q.cited_units(v)}==inventory for v in variants)
        prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        actual=prompt+row['generation_token_ids'];assert actual[-1] in STOPS
        paths=[prompt+tok(v,add_special_tokens=False)['input_ids']+[151645] for v in variants]
        n=min(common_prefix(actual,p) for p in paths)
        if not len(prompt)<=n<min(map(len,paths)) or len(actual)-n>128 or max(len(p)-n for p in paths)>32:
            skips.append({'id':row['id'],'reason':'complete_span_outside_bounded_pilot'});continue
        k=d['generation_prefix_tokens'];prefix=tok.decode(row['generation_token_ids'][:k],skip_special_tokens=True).strip()
        parsed=q.cited_units(prefix.rstrip().rstrip(',').rstrip())
        if parsed is None:skips.append({'id':row['id'],'reason':'native_prefix_parse'});continue
        task='stop' if {u['key'] for u in parsed}==inventory else 'continue'
        assert actual[n] not in {p[n] for p in paths}
        groups[task].append({'id':row['id'],'task':task,'prompt_tokens':len(prompt),'shared_tokens':n,
            'actual_ids':actual,'candidate_paths':paths,'candidate_texts':variants,
            'inventory':[{'name':u['name'],'refs':list(u['refs'])} for u in edit['units']],
            'retention_logit_positions':buffer_positions(len(prompt),n,2),
            'positive_suffix_lengths':[len(p)-n for p in paths],'negative_suffix_length':len(actual)-n,
            'selected_teacher_natural_eos':True,'closure_certificate':False})
    selected=[]
    for task,rows in groups.items():
        selected.extend(sorted(rows,key=lambda r:hashlib.sha256((r['id']+'gemini').encode()).hexdigest())[:8])
    assert not RUN.exists();RUN.mkdir();q.write_rows(RUN/'candidates.jsonl',selected)
    audit={'source_sha256':q.sha(SOURCE),'candidate_counts':{k:len(v) for k,v in groups.items()},
        'selected_counts':{k:sum(r['task']==k for r in selected) for k in groups},'rows':len(selected),
        'gate':len(selected)>=8 and sum(r['task']=='continue' for r in selected)>=4,
        'skips':skips,'all_observed_inventory_preserved':True,'invented_cap_EOS':0,
        'warning':'Natural teacher EOS is behavior, not global semantic closure. Fixed small train-side slice, not confirmation.',
        'code_sha256':q.sha(__file__)}
    q.dump(RUN/'compile_audit.json',audit)
    q.dump(RUN/'protocol.json',{'arms':ARMS,'starting_adapter_sha256':q.sha(q.ANCHOR/'adapter_model.safetensors'),
        'epochs':2,'lr':1e-5,'grad_accum':4,'kl_alpha':1.,'kl_target':.005,'buffer_tokens':2,
        'retention_scope':'Fresh exact common-prefix response-logit coordinates before the last2 prefix tokens; not old re-tokenized full mask.',
        'positive_loss':'Ordinary full-suffix token mean CE, including semantic EOS with same per-token coefficient.',
        'pair_loss':'relu(2-(mean positive logp-mean actual negative logp)) + terminal best-nonEOS hinge / positive_length.',
        'reference_variant_selection':'Frozen anchor maximum complete-path SUM logp among bare/period/newline equivalent closures.',
        'control':'Same positive/negative forwards and backwards, extra pair gradients0. Gradient magnitudes differ.',
        'negative_paths':'Actual full naturally ended teacher suffix, not fabricated truncated EOS.',
        'span_bound':'Positive<=32, negative<=128; preserve complete closure rather than force a5-20token cut.',
        'not_standard_DPO':True,'diagnostic':'Unforced greedy continuation from exact real shared prefix; graph inventory + no repeats + actual EOS.',
        'no_dev_or_sealed_predictions':True,'no_claim_40_of72':'Only8 old native stop states meet conservative behavioral-inventory criterion.',
        'code_sha256':q.sha(__file__)})
    print(json.dumps({k:v for k,v in audit.items() if k!='skips'},indent=2),flush=True)


def model_for(arm=None,train=False):
    original=q.RUN
    try:
        q.RUN=RUN
        return q.load_model(arm or 'anchor',train=train)
    finally:q.RUN=original


def outputs(model,row,path):
    import torch
    import torch.nn.functional as F
    x=torch.tensor(path,device=model.get_input_embeddings().weight.device)[None]
    logits=model(input_ids=x,use_cache=False).logits[0];n=row['shared_tokens']
    lp=F.log_softmax(logits[n-1:-1].float(),-1);targets=x[0,n:].to(lp.device)
    picked=lp.gather(-1,targets[:,None]).squeeze(-1)
    assert path[-1] in STOPS
    picked=torch.cat([picked[:-1],torch.logsumexp(lp[-1,STOPS],0)[None]])
    hinge,rejected=terminal_competition(lp[-1])
    positions=row['retention_logit_positions']
    prefix=F.log_softmax(logits[positions].float(),-1) if positions else None
    return picked,hinge,prefix


def cache():
    import torch
    assert json.loads((RUN/'compile_audit.json').read_text())['gate']
    out=RUN/'reference';out.mkdir();model,tok,_=model_for();model.eval();records=[]
    with torch.inference_mode():
        for row in q.read(RUN/'candidates.jsonl'):
            values=[]
            for path in row['candidate_paths']:
                picked,hinge,prefix=outputs(model,row,path);values.append(float(picked.sum()));del picked,hinge,prefix
            chosen=max(range(len(values)),key=lambda i:values[i])
            record=dict(row,positive_ids=row['candidate_paths'][chosen],reference_variant_scores=values,selected_variant=chosen)
            picked,_,prefix=outputs(model,row,record['positive_ids'])
            if prefix is not None:torch.save(prefix.cpu(),out/(row['id']+'.pt'))
            records.append(record);del picked,prefix
    q.write_rows(RUN/'records.jsonl',records)
    q.dump(out/'manifest.json',{'rows':len(records),'code_sha256':q.sha(__file__),
        'records_sha256':q.sha(RUN/'records.jsonl'),'starting_adapter_sha256':q.sha(q.ANCHOR/'adapter_model.safetensors')})


def train(arm):
    import torch
    import torch.nn.functional as F
    model,tok,_=model_for(train=True);model.train();rows=q.read(RUN/'records.jsonl')
    out=RUN/arm;out.mkdir();opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-5)
    opt.zero_grad(set_to_none=True);history=[];orders=[];begin=time.time()
    for epoch in range(2):
        order=sorted(range(len(rows)),key=lambda i:hashlib.sha256((rows[i]['id']+str(20261014+epoch)).encode()).hexdigest())
        orders.extend(rows[i]['id'] for i in order)
        for visit,i in enumerate(order,1):
            row=rows[i];good,terminal,prefix=outputs(model,row,row['positive_ids']);ce=-good.mean()
            if prefix is not None:
                anchor=torch.load(RUN/'reference'/(row['id']+'.pt'),map_location=prefix.device,weights_only=True)
                kl=(anchor.exp()*(anchor-prefix)).sum(-1).mean();safety=F.relu(kl-.005)
                initial_error=float(kl.detach())
                if epoch==0 and visit==1:assert abs(initial_error)<1e-5,initial_error
                del anchor,prefix
            else:safety=ce*0;initial_error=0.
            ((ce+safety)/4).backward(retain_graph=True)
            bad,unused,unused_prefix=outputs(model,row,row['actual_ids']);del unused,unused_prefix
            pair=F.relu(2.-(good.mean()-bad.mean()))+terminal/len(good)
            ((pair if arm=='span_pair' else pair*0)/4).backward()
            cv=float(ce.detach());pv=float(pair.detach());del good,bad,terminal,ce,pair,safety
            if visit%4==0 or visit==len(order):
                if visit%4:
                    for p in model.parameters():
                        if p.grad is not None:p.grad.mul_(4/(visit%4))
                gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert torch.isfinite(torch.tensor(gn))
                opt.step();opt.zero_grad(set_to_none=True)
                record={'arm':arm,'epoch':epoch+1,'visit':visit,'updates':len(history)+1,
                    'ce':cv,'pair_loss':pv,'prefix_kl':initial_error,'grad_norm':gn,'seconds':time.time()-begin}
                history.append(record);q.dump(out/'status.json',dict(record,state='TRAINING'));print(json.dumps(record),flush=True)
    model.save_pretrained(out/'adapter')
    q.dump(out/'manifest.json',{'arm':arm,'rows':len(rows),'updates':len(history),'history':history,
        'order_sha256':hashlib.sha256(json.dumps(orders).encode()).hexdigest(),'data_sha256':q.sha(RUN/'records.jsonl'),
        'code_sha256':q.sha(__file__),'adapter_sha256':q.sha(out/'adapter/adapter_model.safetensors')})
    q.dump(out/'status.json',{'state':'COMPLETE','updates':len(history)})


def diagnose(arm):
    import torch
    model,tok,_=model_for(None if arm=='anchor' else arm);model.eval();records=[]
    with torch.inference_mode():
        for row in q.read(RUN/'records.jsonl'):
            n=row['shared_tokens'];context=row['positive_ids'][:n]
            x=torch.tensor(context,device=model.get_input_embeddings().weight.device)[None]
            logits=model(input_ids=x,use_cache=False).logits[0,-1]
            allowed={p[n] for p in row['candidate_paths']}
            model.config.use_cache=True
            generated=model.generate(input_ids=x,max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)[0,n:].tolist()
            model.config.use_cache=False
            text=tok.decode(context[row['prompt_tokens']:]+generated,skip_special_tokens=True).strip()
            parsed=q.cited_units(text)
            inventory={(u['name'].casefold(),tuple(u['refs'])) for u in row['inventory']}
            got={u['key'] for u in parsed} if parsed else None
            eos=bool(generated and generated[-1] in STOPS)
            success=bool(parsed and got==inventory and len(parsed)==len(got) and eos)
            records.append({'id':row['id'],'task':row['task'],'first_allowed_argmax':int(logits.argmax()) in allowed,
                'output':text,'generated_ids':generated,'actual_eos':eos,'clean_inventory_equivalent_eos':success})
    out=RUN/'diagnostic'/arm;out.mkdir(parents=True);q.write_rows(out/'predictions.jsonl',records)
    report={'arm':arm,'n':len(records),'first_allowed_argmax':sum(r['first_allowed_argmax'] for r in records),
        'unforced_clean_equivalent_eos':sum(r['clean_inventory_equivalent_eos'] for r in records),
        'by_task':{t:{'n':sum(r['task']==t for r in records),'success':sum(r['task']==t and r['clean_inventory_equivalent_eos'] for r in records)} for t in ['stop','continue']},
        'training_side_only':True,'semantic_closure_claim':False,'code_sha256':q.sha(__file__)}
    q.dump(out/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


def pipeline(defer_pid):
    deferred=False
    try:
        compile_data()
        if not json.loads((RUN/'compile_audit.json').read_text())['gate']:
            q.dump(RUN/'pipeline_status.json',{'state':'NOT_STARTED_FAILED_DATA_GATE'});return
        if defer_pid:
            command=Path('/proc')/str(defer_pid)/'cmdline'
            assert command.stat().st_uid==os.getuid()
            assert 'qwen_equal_curriculum_train_pipeline_20261014.py' in command.read_text()
            from qwen_equal_curriculum_20261014 import RUN as CURRICULUM
            assert json.loads((CURRICULUM/'training_pipeline_status.json').read_text())['state']=='WAITING_FOR_COLLECTION'
            os.kill(defer_pid,signal.SIGSTOP);deferred=True
            q.dump(RUN/'resource_schedule.json',{'deferred_own_supervisor_pid':defer_pid,'state':'TEMPORARILY_DEFERRED','data_collection_continues':True})
        audit=q.ROOT/'runs/qwen_gemini_audit_20261014/probe_status.json'
        while not audit.exists() or json.loads(audit.read_text())['state']!='COMPLETE':time.sleep(5)
        logroot=q.ROOT/'logs/qwen_gemini_span_20261014';logroot.mkdir()
        tasks=[('reference',['--cache']),('anchor_diagnostic',['--diagnose','anchor'])]
        for arm in ARMS:tasks += [(arm,['--train',arm]),(arm+'_diagnostic',['--diagnose',arm])]
        for name,args in tasks:
            q.dump(RUN/'pipeline_status.json',{'state':'RUNNING','stage':name})
            with (logroot/(name+'.log')).open('x') as f:
                subprocess.run([sys.executable,__file__,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
        q.dump(RUN/'pipeline_status.json',{'state':'COMPLETE','no_dev_or_sealed_predictions':True})
    except Exception:
        if RUN.exists():q.dump(RUN/'pipeline_status.json',{'state':'FAILED'})
        raise
    finally:
        if deferred:
            os.kill(defer_pid,signal.SIGCONT)
            q.dump(RUN/'resource_schedule.json',{'deferred_own_supervisor_pid':defer_pid,'state':'RESUMED','data_collection_continues':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pipeline',action='store_true');ap.add_argument('--defer-pid',type=int)
    ap.add_argument('--cache',action='store_true');ap.add_argument('--train',choices=ARMS);ap.add_argument('--diagnose',choices=['anchor']+ARMS);a=ap.parse_args()
    if a.pipeline:pipeline(a.defer_pid)
    elif a.cache:cache()
    elif a.train:train(a.train)
    else:diagnose(a.diagnose)

"""Fresh identical LoRA, Base selective KL, CE vs asymmetric BC hinge."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
import time
import qwen_equal_graph_20261014 as q
from qwen_base_entity_abc_data_20261015 import RUN,STOPS,SEED
from qwen_base_entity_abc_eval_20261015 import load
from entity_abc_contracts_20261015 import validate_label,parse_closed,inventory,supported_unit


def forward(model,row):
    import torch
    import torch.nn.functional as F
    ids=torch.tensor(row['positive_ids'],device=model.get_input_embeddings().weight.device)[None]
    logits=model(input_ids=ids,use_cache=False).logits[0];start=row['shared_tokens']
    lp=F.log_softmax(logits[start-1:-1].float(),-1)
    targets=ids[0,start:].to(lp.device);picked=lp.gather(-1,targets[:,None]).squeeze(-1)
    if row['positive_has_EOS']:picked=torch.cat([picked[:-1],torch.logsumexp(lp[-1,STOPS],0)[None]])
    pair=lp.sum()*0
    for site in row['pair_sites']:
        if not site['rejected']:continue
        pos=torch.logsumexp(lp[site['index'],site['allowed']],0)
        neg=torch.logsumexp(lp[site['index'],site['rejected']],0)
        pair=pair+F.relu(2.-(pos-neg))
    if row['pair_sites']:pair=pair/len(row['pair_sites'])
    positions=row['retention_logit_positions']
    prefix=F.log_softmax(logits[positions].float(),-1) if positions else None
    return picked,pair,prefix


def cache():
    import torch
    model,tok,mapping=load('base');model.eval()
    rows=q.read(RUN/'compiled/records.jsonl');folder=RUN/'reference';folder.mkdir()
    results=[];start=time.time()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            picked,pair,prefix=forward(model,row)
            if prefix is not None:torch.save(prefix.cpu(),folder/(row['id']+'.pt'))
            results.append({'id':row['id'],'type':row['type'],'mean_positive_logp':float(picked.mean()),
                'pair_hinge':float(pair),'KL_positions':len(row['retention_logit_positions']),
                'cache_sha256':q.sha(folder/(row['id']+'.pt')) if prefix is not None else None})
            del picked,pair,prefix
            q.dump(folder/'status.json',{'state':'CACHING','completed':i+1,'of':len(rows),'seconds':time.time()-start})
    q.write_rows(folder/'scores.jsonl',results)
    q.dump(folder/'manifest.json',{'reference':'Official Base only','adapter_loaded':False,
        'rows':len(rows),'records_sha256':q.sha(RUN/'compiled/records.jsonl'),'script_sha256':q.sha(__file__)})
    q.dump(folder/'status.json',{'state':'COMPLETE','rows':len(rows)})


def adapter_init_hash(model):
    import hashlib
    value=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        if p.requires_grad:value.update(name.encode());value.update(p.detach().float().cpu().numpy().tobytes())
    return value.hexdigest()


def train(arm):
    import torch
    import torch.nn.functional as F
    rows=q.read(RUN/'compiled/records.jsonl');assert all(any(r['type']==t for r in rows) for t in ['A','B','C'])
    model,tok,mapping=load(arm,train=True);model.train();folder=RUN/arm;folder.mkdir()
    init=adapter_init_hash(model)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=5e-6)
    opt.zero_grad(set_to_none=True);orders=[];history=[];metrics=[];start=time.time()
    q.dump(folder/'status.json',{'state':'TRAINING'})
    for epoch in range(2):
        order=sorted(rows,key=lambda r:hashlib.sha256((str(SEED+epoch)+r['id']).encode()).hexdigest())
        orders.extend(r['id'] for r in order)
        for visit,row in enumerate(order,1):
            picked,pair,prefix=forward(model,row);ce=-picked.mean()
            if prefix is not None:
                anchor=torch.load(RUN/'reference'/(row['id']+'.pt'),map_location=prefix.device,weights_only=True)
                kl=(anchor.exp()*(anchor-prefix)).sum(-1).mean();safety=F.relu(kl-.005)
                if epoch==0 and visit==1:assert abs(float(kl.detach()))<1e-5, 'Fresh LoRA is not equivalent to Base reference'
                del anchor
                prefix=None
            else:kl=ce*0;safety=ce*0
            factor=1. if arm=='base_pair' and row['type'] in ['B','C'] else 0.
            loss=ce+safety+factor*pair
            assert torch.isfinite(loss)
            (loss/8).backward()
            record={'id':row['id'],'type':row['type'],'epoch':epoch+1,'visit':visit,
                'ce':float(ce.detach()),'pair_hinge_computed_both_arms':float(pair.detach()),
                'pair_weight':factor,'selective_kl':float(kl.detach())}
            metrics.append(record)
            del picked,pair,prefix,ce,kl,safety,loss
            if visit%8==0 or visit==len(order):
                if visit%8:
                    for p in model.parameters():
                        if p.grad is not None:p.grad.mul_(8/(visit%8))
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert torch.isfinite(torch.tensor(norm))
                opt.step();opt.zero_grad(set_to_none=True)
                update=dict(record,update=len(history)+1,grad_norm=norm,seconds=time.time()-start)
                history.append(update);q.dump(folder/'status.json',dict(update,state='TRAINING'));print(json.dumps(update),flush=True)
    model.save_pretrained(folder/'adapter')
    q.write_rows(folder/'visit_history.jsonl',metrics)
    q.dump(folder/'manifest.json',{'arm':arm,'starting_model':q.MODEL,'old_Anchor_loaded':False,
        'reference':'Official Qwen3-14B','initial_trainable_weights_sha256':init,
        'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'data_sha256':q.sha(RUN/'compiled/records.jsonl'),'order_sha256':hashlib.sha256(json.dumps(orders).encode()).hexdigest(),
        'visits':len(metrics),'updates':len(history),'epochs':2,'gradient_accumulation':8,
        'A_contrast_weight':0.,'BC_contrast_weight':1. if arm=='base_pair' else 0.,
        'same_forward_computation':True,'script_sha256':q.sha(__file__),
        'adapter_sha256':q.sha(folder/'adapter/adapter_model.safetensors'),'history':history})
    q.dump(folder/'status.json',{'state':'COMPLETE','updates':len(history)})


def diagnose(which):
    import torch
    model,tok,mapping=load(which);model.eval()
    rows=q.read(RUN/'compiled/records.jsonl')
    inputs={r['id']:r for r in q.read(RUN/'training_source350_snapshot.jsonl')}
    results=[];folder=RUN/'diagnostic'/which;folder.mkdir(parents=True)
    start=time.time()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            validate_label(row);context=row['positive_ids'][:row['shared_tokens']]
            x=torch.tensor(context,device=model.get_input_embeddings().weight.device)[None]
            top=int(model(input_ids=x,use_cache=False).logits[0,-1].argmax())
            ids=model.generate(input_ids=x,max_new_tokens=48,do_sample=False,pad_token_id=tok.eos_token_id)[0,len(context):].tolist()
            body=tok.decode(context[row['prompt_tokens']:]+ids,skip_special_tokens=True)
            parsed=parse_closed(body);entities=inventory(inputs[row['question_id']]);success=False
            if parsed:
                units,tail=parsed
                if row['type'] in ['A','B']:
                    n=row['past_unit_count']
                    success=bool(len(units)>n and supported_unit(units[n],entities)==row['target_entity'])
                else:
                    success=bool(ids and ids[-1] in STOPS and len(units)==row['past_unit_count'] and tail.strip() in ['', '.'])
            results.append({'id':row['id'],'type':row['type'],'output':body,'generated_ids':ids,
                'first_positive_token_argmax':top==row['positive_ids'][row['shared_tokens']],
                'conditional_prefix_success':success,'ended_by_EOS':bool(ids and ids[-1] in STOPS),
                'diagnostic_cap48':bool(len(ids)==48 and ids[-1] not in STOPS)})
            q.dump(folder/'status.json',{'state':'DIAGNOSING','completed':i+1,'of':len(rows),'seconds':time.time()-start})
    q.write_rows(folder/'predictions.jsonl',results)
    summary={'model':which,'by_type':{t:{'n':sum(r['type']==t for r in results),
        'first_positive_argmax':sum(r['type']==t and r['first_positive_token_argmax'] for r in results),
        'conditional_prefix_success':sum(r['type']==t and r['conditional_prefix_success'] for r in results)} for t in ['A','B','C']},
        'training_side_all_selected_rows':True,'offpolicy_B_C':True,'max_diagnostic_new_tokens':48,
        'not_full_prompt_success':True,'lexical_not_semantic_success':True,'script_sha256':q.sha(__file__)}
    q.dump(folder/'summary.json',summary);q.dump(folder/'status.json',{'state':'COMPLETE','rows':len(rows)})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--cache',action='store_true');ap.add_argument('--train',choices=['base_ce','base_pair']);ap.add_argument('--diagnose',choices=['base','base_ce','base_pair']);a=ap.parse_args()
    if a.cache:cache()
    elif a.train:train(a.train)
    else:diagnose(a.diagnose)

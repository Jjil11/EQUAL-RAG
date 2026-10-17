"""Frozen three-seed confirmation and matched single-component ablations.

No sealed data, inference rules, merge, quality gate, or dev-based selection.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import qwen_pair_aonly_20261015 as prev
from qwen_tail_guard_scan_20261015 import free_generate

v3=prev.v3;c=v3.core;q=c.q
SOURCE=prev.SOURCE
AONLY=prev.RUN
RUN=q.ROOT/'runs/qwen_seed_ablation_v2_20261015'
SEEDS=[20261015,20261118,20261229]
CONFIGS={'ce':0.,'a025':.25,'a050':.50}
TASKS={}
for seed in SEEDS[1:]:
    for config,beta in CONFIGS.items():
        TASKS[f'{config}_seed{seed}']={'config':config,'beta':beta,'seed':seed,'variant':'full'}
for variant in ['no_guard','no_stop','no_bpe','stop_on']:
    for config,beta in CONFIGS.items():
        TASKS[f'{config}_{variant}']={'config':config,'beta':beta,'seed':SEEDS[0],'variant':variant}


def order(data,seed,salt='',visits=240):
    result=[];epoch=0
    while len(result)<visits:
        result+=sorted(data,key=lambda r:c.digest(str(seed+epoch)+salt+r['id']))
        epoch+=1
    return result[:visits]


def no_bpe_record(row):
    r=copy.deepcopy(row)
    if r['type']!='A':return r
    root=r['shared_tokens']
    assert r['actual_ids'][-2:]==[60,151645] and root==len(r['actual_ids'])-2
    assert r['positive_ids'][root]==1125
    # Exactly the same decoded characters, but freeze the observed bare ']' and
    # append comma separately instead of rolling back/jointly tokenizing '],'.
    r['positive_ids']=r['actual_ids'][:-1]+[11]+r['positive_ids'][root+1:]
    r['shared_tokens']=r['repair_root_tokens']=root+1
    r['C1_root_tokens']+=1
    r['generated_prefix_retokenized']=False
    r['origin']='same_native_EOS_prefix_frozen_no_boundary_rollback'
    for site in r['pair_sites']:
        if site['group']=='action':
            assert site['index']==0
            site['allowed']=[11];site['rejected']=list(c.STOPS)
    assert r['positive_ids'][:root+1]==r['actual_ids'][:-1]
    return r


def prepare():
    RUN.mkdir(exist_ok=False)
    data=RUN/'data';data.mkdir()
    for name in ['compiled','cached_eval']:
        shutil.copytree(SOURCE/name,data/name)
    for name in ['eval69.jsonl','training_source350_snapshot.jsonl']:
        shutil.copy2(SOURCE/name,data/name)
    prev.link_tree(SOURCE/'reference',data/'reference')
    nb=RUN/'data_no_bpe';nb.mkdir();(nb/'compiled').mkdir()
    for name in ['cached_eval','reference']:
        prev.link_tree(data/name,nb/name)
    for name in ['eval69.jsonl','training_source350_snapshot.jsonl']:
        shutil.copy2(data/name,nb/name)
    shutil.copy2(data/'compiled/native_retention.jsonl',nb/'compiled/native_retention.jsonl')
    original=q.read(data/'compiled/records.jsonl');changed=[no_bpe_record(r) for r in original]
    q.write_rows(nb/'compiled/records.jsonl',changed)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True)
    for before,after in zip(original,changed):
        assert tok.decode(before['positive_ids'])==tok.decode(after['positive_ids'])
        assert before['positive_text']==after['positive_text']
        assert before['state']==after['state'] and before['retention_logit_positions']==after['retention_logit_positions']
    p=json.loads((SOURCE/'protocol.json').read_text())
    p.update(maximum_steps=60,no_training_gate=True,no_early_stopping=True,
        independent_first_gate_stopping_both_arms=False,no_prior_dev_generation=False,
        seeds=SEEDS,configs=CONFIGS,tasks=TASKS,merge_called=False,max_guard=False,
        inference_rules=False,sealed311_reads=0,source_protocol_sha256=q.sha(SOURCE/'protocol.json'),
        no_stop_design='zero C1/C2 CE; same A/aux slots and denominator; C2 slots retention-only',
        stop_on_design='same A/citation weights beta/0.5beta, add original terminal weight 2',
        no_bpe_design='same decoded targets; native bare ] frozen, separate comma token 11, no rollback/joint retokenization',
        bootstrap_design='mean seed-level per-question sufficient statistics, common paired question draws; not ensemble text',
        no_hyperparameter_selection=True,script_sha256=q.sha(__file__))
    q.dump(RUN/'protocol.json',p)
    q.dump(RUN/'status.json',{'state':'PREPARED','planned_new_jobs':len(TASKS),'sealed311_reads':0})


def weights(p,spec):
    return dict(p,pair_weights={'action':spec['beta'],'citation':.5*spec['beta'],
        'terminal':2. if spec['variant']=='stop_on' else 0.})


def supervised(model,row,p,no_stop=False):
    if not no_stop:return v3.supervised(model,row,p)
    import torch
    import torch.nn.functional as F
    n=row['shared_tokens']
    x=torch.tensor(row['positive_ids'],device=model.get_input_embeddings().weight.device)[None]
    logits=model(input_ids=x,use_cache=False).logits[0]
    lp=F.log_softmax(logits[n-1:-1].float(),-1)
    prefix=F.log_softmax(logits[row['retention_logit_positions']].float(),-1)
    targets=torch.tensor(row['positive_ids'][n:],device=lp.device)
    picked=lp.gather(1,targets[:,None]).squeeze(1)
    picked=torch.cat([picked[:-1],torch.logsumexp(lp[-1,c.STOPS],0)[None]])
    zero=picked.sum()*0
    if row['type']=='A':
        mask=torch.ones(len(picked),device=lp.device,dtype=torch.bool)
        mask[row['terminal_relative_positions']]=False
        first=-picked[0]
        # Do not redistribute removed stop-token CE onto remaining A tokens.
        span=-picked[mask].sum()/len(picked)
        semantic=-picked[row['semantic_positive_positions']].mean()
        ce=first+span+p['semantic_positive_CE_weight']*semantic
    else:ce=first=span=zero
    groups={}
    for s in row['pair_sites']:
        if s['group']=='terminal':continue
        pos=torch.logsumexp(lp[s['index'],s['allowed']],0)
        neg=torch.logsumexp(lp[s['index'],s['rejected']],0)
        groups.setdefault(s['group'],[]).append(F.relu(p['pair_margin']-(pos-neg)))
    hinge=sum((p['pair_weights'][k]*torch.stack(v).mean() for k,v in groups.items()),zero)
    return ce,hinge,first,span,zero,prefix


def load_train(seed):
    import torch
    from peft import LoraConfig,get_peft_model
    model,tok,placement=q.load_model('base')
    q.seed_all(seed)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable();model.enable_input_require_grads()
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
    model.config.use_cache=False
    return model,tok,placement


def train(task):
    import torch
    spec=TASKS[task];folder=RUN/task;folder.mkdir(exist_ok=False)
    data=RUN/('data_no_bpe' if spec['variant']=='no_bpe' else 'data')
    c.RUN=c.ev.RUN=data
    p=weights(json.loads((RUN/'protocol.json').read_text()),spec)
    rows=q.read(data/'compiled/records.jsonl');native=q.read(data/'compiled/native_retention.jsonl')
    bynative={r['id']:r for r in native}
    sup=order(rows,spec['seed']);aux=order(native,spec['seed'],'retention')
    old=json.loads((SOURCE/'base_ce/manifest.json').read_text())
    sh=c.digest(json.dumps([r['id'] for r in sup]));ah=c.digest(json.dumps([r['id'] for r in aux]))
    if spec['seed']==SEEDS[0]:assert sh==old['order_sha256'] and ah==old['aux_order_sha256']
    model,tok,_=load_train(spec['seed']);init=c.adapter_init_hash(model)
    if spec['seed']==SEEDS[0]:assert init==old['initial_trainable_weights_sha256']
    else:assert init!=old['initial_trainable_weights_sha256'] and sh!=old['order_sha256']
    optimizer=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=p['lr'])
    optimizer.zero_grad(set_to_none=True);started=time.time();keep=spec['variant']!='no_guard'
    with (folder/'visits.jsonl').open('x') as stream:
        for step in range(1,61):
            model.train()
            for j in range(4):
                r=sup[(step-1)*4+j]
                ce,hinge,first,span,terminal,lp=supervised(model,r,p,spec['variant']=='no_stop')
                if keep:
                    own=bynative[r['native_id']]
                    assert own['positions']==r['retention_logit_positions']
                    assert all(r['positive_ids'][:k+1]==own['actual_ids'][:k+1] for k in own['positions'])
                    ref=torch.load(data/'reference'/(own['id']+'.pt'),map_location=lp.device,weights_only=True)
                    penalty,kl,fid,flip=v3.retention_loss(lp,ref,p)
                    del ref
                else:penalty=kl=fid=flip=ce*0
                loss=ce+hinge+penalty;assert torch.isfinite(loss);(loss/4).backward()
                item={'step':step,'id':r['id'],'type':r['type'],'CE':float(ce.detach()),'hinge':float(hinge.detach()),
                    'KL':float(kl.detach()),'fidelity':float(fid.detach()),'flip':float(flip.detach()),
                    'supervised_CE_active':not(spec['variant']=='no_stop' and r['type']=='C')}
                del ce,hinge,first,span,terminal,lp,penalty,kl,fid,flip,loss
                extra=aux[(step-1)*4+j];item['aux_id']=extra['id']
                if keep:
                    lp=c.get_lp(model,extra['actual_ids'],extra['positions'])
                    ref=torch.load(data/'reference'/(extra['id']+'.pt'),map_location=lp.device,weights_only=True)
                    penalty,kl,fid,flip=v3.retention_loss(lp,ref,p)
                    (.5*penalty/4).backward();item['aux_KL']=float(kl.detach())
                    del lp,ref,penalty,kl,fid,flip
                stream.write(json.dumps(item)+'\n');stream.flush()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert torch.isfinite(torch.tensor(norm))
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            q.dump(folder/'status.json',{'state':'TRAINING_FIXED_60','step':step,'seconds':time.time()-started})
            print(json.dumps(dict(item,grad_norm=norm)),flush=True)
    model.save_pretrained(folder/'adapter')
    q.dump(folder/'manifest.json',{'spec':spec,'updates':60,'supervised_slots':240,'A_visits':sum(r['type']=='A' for r in sup),
        'C2_slots':sum(r['type']=='C' for r in sup),'aux_slots':240,'aux_retention_active':keep,
        'initial_trainable_weights_sha256':init,'final_trainable_weights_sha256':c.adapter_init_hash(model),
        'order_sha256':sh,'aux_order_sha256':ah,'data_sha256':q.sha(data/'compiled/records.jsonl'),
        'retention_sha256':q.sha(data/'compiled/native_retention.jsonl'),'adapter_sha256':q.sha(folder/'adapter/adapter_model.safetensors'),
        'pair_weights':p['pair_weights'],'merge_called':False,'no_quality_gate':True,'max_guard':False,'script_sha256':q.sha(__file__)})
    del optimizer,model;gc.collect();torch.cuda.empty_cache()


def profile(task):
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    spec=TASKS[task];data=RUN/('data_no_bpe' if spec['variant']=='no_bpe' else 'data')
    c.RUN=c.ev.RUN=data;folder=RUN/task/'profile';folder.mkdir()
    model,tok,placement=q.load_model('base')
    model=PeftModel.from_pretrained(model,str(RUN/task/'adapter'),is_trainable=False)
    layers=[m for m in model.modules() if isinstance(m,LoraLayer)]
    assert layers and not any(m.merged for m in layers)
    q.dump(folder/'manifest.json',{'LoRA_layers':len(layers),'all_unmerged':True,'merge_called':False,
        'placement':placement,'greedy':True,'max_new_tokens':256,'inference_rules':False})
    fidelity=c.fidelity_probe(model,json.loads((data/'reference/index.json').read_text()),folder/'fidelity')
    diagnostic=c.diagnose(model,tok,task,folder/'diagnostic')
    records={r['question_id']:r for r in q.read(data/'compiled/records.jsonl') if r['type']=='A'}
    ti={r['id']:r for r in q.read(data/'training_source350_snapshot.jsonl')}
    ar=[{'id':k,'prompt_token_ids':r['actual_ids'][:r['prompt_tokens']],'record':r,'input':ti[k]} for k,r in sorted(records.items())]
    out=free_generate(model,tok,ar,folder/'training_A14',training=True)
    ts={'full_prefix_preserved':sum(r['old_repair_root_reached_exactly'] for r in out),
        'locally_cited':{k:sum(len(r['locally_cited'][k]) for r in out) for k in ['added','lost']}}
    q.dump(folder/'training_A14/summary.json',ts)
    base={r['id']:r for r in q.read(data/'cached_eval/base.jsonl')}
    er=[{'id':r['id'],'prompt_token_ids':base[r['id']]['prompt_token_ids']} for r in q.read(data/'eval69.jsonl')]
    free_generate(model,tok,er,folder/'eval69')
    q.dump(folder/'status.json',{'state':'COMPLETE','fidelity':fidelity,'diagnostic':diagnostic,'training_A14':ts,'eval_predictions':69})
    q.dump(RUN/task/'status.json',{'state':'COMPLETE','step':60,'merge_called':False})


def pipeline():
    prepare()
    pending=list(TASKS);workers={};finished=[];failed=[]
    while pending or workers:
        for gpu in ['0,2','1,3']:
            if gpu in workers or not pending:continue
            task=pending.pop(0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            log=(q.ROOT/'logs'/('qwen_seed_ablation_'+task+'_20261015.log')).open('x')
            process=subprocess.Popen([sys.executable,__file__,'--job',task],cwd=q.ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
            workers[gpu]=(task,process,log)
        for gpu,(task,process,log) in list(workers.items()):
            rc=process.poll()
            if rc is None:continue
            log.close();(finished if rc==0 else failed).append(task);del workers[gpu]
        q.dump(RUN/'status.json',{'state':'RUNNING_FROZEN_QUEUE','running':{g:v[0] for g,v in workers.items()},
            'queued':pending,'finished':finished,'failed':failed,'sealed311_reads':0})
        if pending or workers:time.sleep(10)
    assert not failed,failed
    q.dump(RUN/'status.json',{'state':'ALL_GENERATIONS_COMPLETE','jobs':len(finished),'new_predictions':69*len(finished),'sealed311_reads':0})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pipeline',action='store_true');ap.add_argument('--job',choices=TASKS)
    z=ap.parse_args()
    if z.pipeline:pipeline()
    elif z.job:train(z.job);profile(z.job)

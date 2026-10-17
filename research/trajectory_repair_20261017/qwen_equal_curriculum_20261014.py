"""Qwen-only decoding controls and high-coverage teacher collection.

Identical explicit-format inputs for all models. Fixed sampling seed per input,
not output selection. Gold is used only by offline scoring and compilation.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import qwen_equal_graph_20261014 as q
from qwen_equal_format_controls_20261014 import INSTRUCTION

RUN=q.ROOT/'runs/qwen_equal_curriculum_20261014'
TEACHER=q.ROOT/'runs/qwen_signal_split_20261014/repair_only/adapter'
FMT=q.ROOT/'runs/qwen_equal_graph_format_20261014'
SEED=20261014


def decode_kwargs(mode):
    if mode=='greedy':return {'do_sample':False}
    assert mode=='sample'
    return dict(do_sample=True,temperature=.7,top_p=.8,top_k=20,min_p=0.,repetition_penalty=1.)


def input_seed(row):
    return int(hashlib.sha256((str(SEED)+row['id']).encode()).hexdigest()[:8],16)


def load(which,train=False):
    original_run,original_anchor=q.RUN,q.ANCHOR
    try:
        q.RUN=RUN
        if which=='repair':q.ANCHOR=TEACHER
        return q.load_model('anchor' if which=='repair' else which,train=train)
    finally:q.RUN,q.ANCHOR=original_run,original_anchor


def adapter_path(which):
    return q.ANCHOR if which=='anchor' else TEACHER if which=='repair' else RUN/which/'adapter'


def prepare():
    assert not RUN.exists();RUN.mkdir()
    rows=q.read(FMT/'validation.jsonl');training=q.read(q.SOURCE/'train.jsonl')
    assert len(rows)==48 and len(training)==128
    assert not {r['question_id'] for r in rows}&{r['question_id'] for r in training}
    for r in training:r['prompt']+=INSTRUCTION
    q.write_rows(RUN/'validation.jsonl',rows);q.write_rows(RUN/'train.jsonl',training)
    assert q.sha(RUN/'validation.jsonl')==q.sha(FMT/'validation.jsonl')
    for which in ['base','anchor']:
        destination=RUN/'eval'/(which+'__greedy');destination.parent.mkdir(exist_ok=True)
        shutil.copytree(FMT/'eval'/which,destination)
        q.dump(destination/'reuse_provenance.json',{'source':str(FMT/'eval'/which),
            'source_predictions_sha256':q.sha(FMT/'eval'/which/'predictions.jsonl'),
            'prompt_sha256':q.sha(RUN/'validation.jsonl'),'decoding':'greedy','not_a_new_generation':True})
    q.dump(RUN/'protocol.json',{
        'purpose':'Separate decoder-induced loops, teacher reliability and one-sided coverage curriculum.',
        'primary_condition':'Same explicit-format prompt for all models; non-thinking, BF16, cap256, maxprompt2048.',
        'models':['base','anchor','repair'],'decoders':['greedy','sample'],
        'official_sampling':decode_kwargs('sample'),'official_source':'https://huggingface.co/Qwen/Qwen3-14B',
        'official_greedy_warning_scope':'Thinking mode; we do NOT claim non-thinking greedy is prohibited.',
        'sampling_seed':SEED,'seed_per_input':'SHA256(seed + input id), independent of model and worker order.',
        'validation_sha256':q.sha(RUN/'validation.jsonl'),'train_sha256':q.sha(RUN/'train.jsonl'),
        'teacher_adapter_sha256':q.sha(TEACHER/'adapter_model.safetensors'),
        'anchor_adapter_sha256':q.sha(q.ANCHOR/'adapter_model.safetensors'),
        'development_questions':24,'training_questions':64,'cached_eval_predictions':96,
        'new_decoder_eval_predictions':192,'new_teacher_generations':256,
        'screen':'Each view Q/F1>=anchor greedy-.02, cost<=.9anchor greedy and<=anchor greedy repeat stop; Q/F1>=rule-.02. Compare sampling to matched sampling controls separately.',
        'training_data_selection':'Offline only: prefer natural sampled answer only when all closed greedy units are preserved and censored tail is not lost; otherwise keep greedy teacher. Never select dev outputs.',
        'closure_warning':'Native EOS is a teacher behavior prior, not semantic closure. Capped targets get no EOS.',
        'no_sealed_reads_or_predictions':True,'extra_models':0,'code_sha256':q.sha(__file__)})


def generate(which,mode,phase='eval'):
    import torch
    root=RUN/('collection' if phase=='collect' else 'eval')/(which+'__'+mode)
    root.mkdir(parents=True,exist_ok=False);q.dump(root/'status.json',{'state':'LOADING'})
    try:
        model,tok,mapping=load(which);model.eval()
        source=RUN/('train.jsonl' if phase=='collect' else 'validation.jsonl');rows=q.read(source)
        stops=model.generation_config.eos_token_id;stops=[stops] if isinstance(stops,int) else list(stops)
        q.dump(root/'manifest.json',{'model':which,'mode':mode,'phase':phase,'source_sha256':q.sha(source),
            'adapter_sha256':None if which=='base' else q.sha(adapter_path(which)/'adapter_model.safetensors'),
            'kwargs':decode_kwargs(mode),'stops':stops,'placement':mapping,'code_sha256':q.sha(__file__)})
        begin=time.time();records=[]
        with (root/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode():
            for i,r in enumerate(rows):
                q.seed_all(input_seed(r))
                text=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                enc=tok(text,return_tensors='pt').to(model.get_input_embeddings().weight.device)
                assert len(enc['input_ids'][0])<=2048
                ids=model.generate(**enc,max_new_tokens=256,pad_token_id=tok.eos_token_id,**decode_kwargs(mode))[0,len(enc['input_ids'][0]):].tolist()
                p={'id':r['id'],'question_id':r['question_id'],'view':r['view'],'input_seed':input_seed(r),
                    'prompt_ids':enc['input_ids'][0].tolist(),'generated_ids':ids,'tokens':len(ids),
                    'output':tok.decode(ids,skip_special_tokens=True).strip(),'eos_ids':stops,
                    'ended_by_eos':bool(ids and ids[-1] in stops),'hit_cap_without_eos':len(ids)==256 and ids[-1] not in stops}
                f.write(json.dumps(p,ensure_ascii=False)+'\n');f.flush();records.append(p)
                status={'state':'GENERATING','model':which,'mode':mode,'phase':phase,'completed':i+1,'of':len(rows),'seconds':time.time()-begin}
                q.dump(root/'status.json',status)
                if (i+1)%8==0:print(json.dumps(status),flush=True)
        if phase=='eval':q.score_records(rows,records,root)
        q.dump(root/'status.json',{'state':'COMPLETE','rows':len(rows),'seconds':time.time()-begin})
    except Exception:q.dump(root/'status.json',{'state':'FAILED'});raise


def worker(name):
    tasks=([('base','sample','eval'),('anchor','sample','eval')] if name=='controls' else
        [('repair','greedy','eval'),('repair','sample','eval'),('repair','sample','collect'),('repair','greedy','collect')])
    for which,mode,phase in tasks:
        q.dump(RUN/(name+'_worker_status.json'),{'state':'RUNNING','model':which,'mode':mode,'phase':phase})
        subprocess.run([sys.executable,__file__,'--generate',which,'--mode',mode,'--phase',phase],check=True)
    q.dump(RUN/(name+'_worker_status.json'),{'state':'COMPLETE'})


def pipeline():
    prepare();logroot=q.ROOT/'logs/qwen_equal_curriculum_20261014';logroot.mkdir()
    processes=[];handles=[]
    for name,devices in [('controls','0,2'),('teacher','1,3')]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        log=(logroot/(name+'.log')).open('x');handles.append(log)
        p=subprocess.Popen([sys.executable,__file__,'--worker',name],env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((name,p))
    q.dump(RUN/'pipeline_status.json',{'state':'RUNNING','worker_pids':{n:p.pid for n,p in processes}})
    codes={n:p.wait() for n,p in processes}
    for f in handles:f.close()
    assert all(v==0 for v in codes.values()),codes
    q.dump(RUN/'pipeline_status.json',{'state':'COMPLETE_DECODING_AND_COLLECTION','exit_codes':codes})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pipeline',action='store_true');ap.add_argument('--worker',choices=['controls','teacher'])
    ap.add_argument('--generate');ap.add_argument('--mode',choices=['greedy','sample']);ap.add_argument('--phase',choices=['eval','collect'],default='eval');a=ap.parse_args()
    if a.pipeline:
        try:pipeline()
        except Exception:
            if RUN.exists():q.dump(RUN/'pipeline_status.json',{'state':'FAILED'})
            raise
    elif a.worker:worker(a.worker)
    else:generate(a.generate,a.mode,a.phase)

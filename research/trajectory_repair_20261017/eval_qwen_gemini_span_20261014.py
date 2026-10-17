"""V2 raw-dev transfer; V1 failed its overlap guard before any predictions."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
import subprocess
import sys
import time
import qwen_equal_graph_20261014 as q
import qwen_equal_curriculum_20261014 as c
import qwen_gemini_span_20261014 as m
from qwen_equal_format_controls_20261014 import INSTRUCTION
from score_strict_units_20261010 import norm,score,summarize

CURRICULUM=c.RUN
SNAPSHOT=q.ROOT/'runs/qwen_decision_recovery_20261012/analysis/prospective_dev/data_full_gold_snapshot.jsonl'
MODELS=['base','anchor']+m.ARMS


def prepare():
    assert json.loads((m.RUN/'pipeline_status.json').read_text())['state']=='COMPLETE'
    selected=q.read(m.RUN/'records.jsonl');original=q.read(m.SOURCE)
    train_ids={r['id'] for r in original};train_questions={norm(r['question']) for r in original}
    previous=q.read(CURRICULUM/'validation.jsonl')
    overlap=sorted({r['id'] for r in selected}&{r['question_id'] for r in previous})
    assert overlap
    failure=json.loads((m.RUN/'transfer_status.json').read_text())
    assert failure['state']=='FAILED' and not (m.RUN/'validation.jsonl').exists()
    q.dump(m.RUN/'transfer_v1_failure.json',{'status':failure,'before_any_transfer_predictions':True,
        'overlapping_micro_training_ids':overlap,'failed_code_sha256':q.sha(q.ROOT/'code/eval_qwen_gemini_span_20261014_failed_v1.py')})
    pool=q.read(SNAPSHOT);assert len(pool)==80
    assert not {r['id'] for r in pool}&train_ids
    assert not {norm(r['question']) for r in pool}&train_questions
    rows=sorted(pool,key=lambda r:hashlib.sha256((r['id']+'gemini-transfer-20261014').encode()).hexdigest())[:24]
    for r in rows:r['question_id']=r['id'];r['view']='raw';r['prompt']+=INSTRUCTION
    q.write_rows(m.RUN/'validation.jsonl',rows)
    q.dump(m.RUN/'transfer_protocol.json',{'version':2,'purpose':'Full-question transfer, not train-prefix fit.',
        'models':MODELS,'decoding':'greedy','max_new_tokens':256,'same_explicit_format_prompt':True,
        'rows':24,'questions':24,'views':['raw'],'historically_used_development_pool':True,
        'independent_blind_confirmation':False,
        'selection':'Fixed SHA256(id + gemini-transfer-20261014); no coverage or output filtering.',
        'pool_source':str(SNAPSHOT),'pool_sha256':q.sha(SNAPSHOT),
        'training_pool_id_and_normalized_question_overlap':0,'training_pool_rows':len(original),
        'zero_known_edge_questions':sum(not r['edges'] for r in rows),
        'validation_sha256':q.sha(m.RUN/'validation.jsonl'),'new_transfer_predictions':96,
        'sealed_predictions':0,'no_checkpoint_selection':True,'paired_comparison_family':8,
        'screen':'Raw Q/F1>=anchor-.02 and<=.9anchor cost; Q/F1>=anchor repeat rule-.02 and nonhigher cost. Point screen only.',
        'code_sha256':q.sha(__file__),'helpers_sha256':q.sha(q.ROOT/'code/qwen_equal_curriculum_20261014.py')})


def generate(which):
    import torch
    c.RUN=m.RUN
    root=m.RUN/'eval'/(which+'__greedy');root.mkdir(parents=True,exist_ok=False)
    q.dump(root/'status.json',{'state':'LOADING'})
    try:
        model,tok,mapping=c.load(which);model.eval();rows=q.read(m.RUN/'validation.jsonl')
        stops=model.generation_config.eos_token_id;stops=[stops] if isinstance(stops,int) else list(stops)
        q.dump(root/'manifest.json',{'model':which,'mode':'greedy','source_sha256':q.sha(m.RUN/'validation.jsonl'),
            'adapter_sha256':None if which=='base' else q.sha(c.adapter_path(which)/'adapter_model.safetensors'),
            'kwargs':c.decode_kwargs('greedy'),'stops':stops,'placement':mapping,'code_sha256':q.sha(__file__)})
        begin=time.time();records=[]
        with (root/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode():
            for i,r in enumerate(rows):
                q.seed_all(c.input_seed(r))
                text=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                enc=tok(text,return_tensors='pt').to(model.get_input_embeddings().weight.device)
                assert len(enc['input_ids'][0])<=2048
                ids=model.generate(**enc,max_new_tokens=256,pad_token_id=tok.eos_token_id,**c.decode_kwargs('greedy'))[0,len(enc['input_ids'][0]):].tolist()
                p={'id':r['id'],'question_id':r['question_id'],'view':'raw','input_seed':c.input_seed(r),
                    'prompt_ids':enc['input_ids'][0].tolist(),'generated_ids':ids,'tokens':len(ids),
                    'output':tok.decode(ids,skip_special_tokens=True).strip(),'eos_ids':stops,
                    'ended_by_eos':bool(ids and ids[-1] in stops),'hit_cap_without_eos':len(ids)==256 and ids[-1] not in stops}
                f.write(json.dumps(p,ensure_ascii=False)+'\n');f.flush();records.append(p)
                q.dump(root/'status.json',{'state':'GENERATING','completed':i+1,'of':len(rows),'seconds':time.time()-begin})
                if (i+1)%8==0:print(json.dumps({'model':which,'completed':i+1,'seconds':time.time()-begin}),flush=True)
        scored=[score(r,dict(p,grounded_edges=0)) for r,p in zip(rows,records)]
        q.write_rows(root/'scored.jsonl',scored)
        q.dump(root/'summary.json',dict(summarize(scored),true_cap_hit_rate=sum(p['hit_cap_without_eos'] for p in records)/len(records)))
        q.dump(root/'status.json',{'state':'COMPLETE','rows':len(rows),'seconds':time.time()-begin})
    except Exception:q.dump(root/'status.json',{'state':'FAILED'});raise


def pipeline():
    prepare();q.dump(m.RUN/'transfer_status.json',{'state':'WAITING_FOR_OWN_GPU_WORKER','version':2})
    until=time.time()+10800
    while True:
        state=json.loads((CURRICULUM/'coverage_bridge_worker_status.json').read_text())
        if state['state']=='COMPLETE':break
        assert time.time()<until
        assert json.loads((CURRICULUM/'training_pipeline_status.json').read_text())['state']!='FAILED'
        time.sleep(15)
    for which in MODELS:
        q.dump(m.RUN/'transfer_status.json',{'state':'RUNNING','model':which,'version':2})
        with (q.ROOT/'logs/qwen_gemini_span_20261014'/(which+'_full_prompt_v2.log')).open('x') as log:
            subprocess.run([sys.executable,__file__,'--generate',which],check=True,stdout=log,stderr=subprocess.STDOUT)
    q.dump(m.RUN/'transfer_status.json',{'state':'COMPLETE','new_transfer_predictions':96,'version':2})


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--generate',choices=MODELS);a=ap.parse_args()
    if a.generate:generate(a.generate)
    else:
        try:pipeline()
        except Exception:q.dump(m.RUN/'transfer_status.json',{'state':'FAILED','version':2});raise

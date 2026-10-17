"""Additional raw-question transfer; prior24 overlap ALL historic350 training."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import subprocess
import sys
import time
import qwen_equal_graph_20261014 as q
import qwen_equal_curriculum_20261014 as c
import qwen_gemini_span_20261014 as m
from score_strict_units_20261010 import norm,score,summarize

RUN=c.RUN/'raw_transfer_20261015'
MODELS=['repair','compact_sft','coverage_bridge']


def prepare():
    assert not RUN.exists();RUN.mkdir()
    rows=q.read(m.RUN/'validation.jsonl');old=q.read(m.SOURCE)
    assert len(rows)==24 and all(r['view']=='raw' for r in rows)
    assert not {r['id'] for r in old}&{r['question_id'] for r in rows}
    assert not {norm(r['question']) for r in old}&{norm(r['question']) for r in rows}
    assert not {norm(r['question']) for r in q.read(c.RUN/'train.jsonl')}&{norm(r['question']) for r in rows}
    historical=q.read(c.RUN/'validation.jsonl')
    overlap=len({r['id'] for r in old}&{r['question_id'] for r in historical});assert overlap==24
    q.write_rows(RUN/'validation.jsonl',rows)
    q.dump(RUN/'protocol.json',{'frozen_before_any_new_raw_transfer_predictions':True,
        'purpose':'Actual full-prompt transfer instead of historical-train regression.',
        'models':MODELS,'new_predictions':72,'cached_fresh_comparator_predictions':48,
        'comparator_source':str(m.RUN/'eval'),'all_inputs_identical_to_gemini_v2':True,
        'rows':24,'old350_training_overlap':0,'current64_training_overlap':0,
        'historical24_regression_questions_overlap_old350':24,
        'zero_known_edge_questions':sum(not r['edges'] for r in rows),'sealed_predictions':0,
        'historically_reused_dev_not_blind_confirmation':True,'decoding':'greedy','max_new_tokens':256,
        'screen':'Q/F1 within.02anchor and<=.9anchor cost; within.02anchor repeat rule and nonhighercost.',
        'teacher_secondary':'Q/F1 within.02repair and<=.9repair cost; also compare repair repeat rule separately.',
        'paired_family':13,'seed_count':1,'no_output_or_coverage_filtering':True,
        'validation_sha256':q.sha(RUN/'validation.jsonl'),'code_sha256':q.sha(__file__)})


def generate(which):
    import torch
    root=RUN/'eval'/which;root.mkdir(parents=True,exist_ok=False)
    q.dump(root/'status.json',{'state':'LOADING'})
    try:
        model,tok,mapping=c.load(which);model.eval();rows=q.read(RUN/'validation.jsonl')
        stops=model.generation_config.eos_token_id;stops=[stops] if isinstance(stops,int) else list(stops)
        q.dump(root/'manifest.json',{'model':which,'mode':'greedy','source_sha256':q.sha(RUN/'validation.jsonl'),
            'adapter_sha256':q.sha(c.adapter_path(which)/'adapter_model.safetensors'),
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
        sc=[score(r,dict(p,grounded_edges=0)) for r,p in zip(rows,records)]
        q.write_rows(root/'scored.jsonl',sc)
        q.dump(root/'summary.json',dict(summarize(sc),true_cap_hit_rate=sum(p['hit_cap_without_eos'] for p in records)/len(records)))
        q.dump(root/'status.json',{'state':'COMPLETE','rows':len(rows),'seconds':time.time()-begin})
    except Exception:q.dump(root/'status.json',{'state':'FAILED'});raise


def pipeline():
    prepare();q.dump(RUN/'status.json',{'state':'WAITING_FOR_OWN_WORKERS'})
    until=time.time()+14400
    while True:
        a=json.loads((m.RUN/'transfer_status.json').read_text())['state']
        b=json.loads((c.RUN/'training_pipeline_status.json').read_text())['state']
        assert a!='FAILED' and b!='FAILED' and time.time()<until
        if a==b=='COMPLETE':break
        time.sleep(15)
    for which in MODELS:
        q.dump(RUN/'status.json',{'state':'RUNNING','model':which})
        with (q.ROOT/'logs/qwen_equal_curriculum_20261014'/(which+'_raw_transfer_20261015.log')).open('x') as log:
            subprocess.run([sys.executable,__file__,'--generate',which],stdout=log,stderr=subprocess.STDOUT,check=True)
    q.dump(RUN/'status.json',{'state':'COMPLETE','new_predictions':72})


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--generate',choices=MODELS);a=ap.parse_args()
    if a.generate:generate(a.generate)
    else:
        try:pipeline()
        except Exception:q.dump(RUN/'status.json',{'state':'FAILED'});raise

"""Read-only training-state branch and true greedy-EOS diagnostic.

Teacher-forced states only: these are not independent evaluation samples.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from completion_pairs_20261010 import ROOT,ROUND,compiled_pair
from train_verified_boundary_20261010 import eos_ids


def branch(model,ids,n,stops):
    logits=model(input_ids=ids[None].cuda(),use_cache=False).logits[0]
    lp=torch.log_softmax(logits[n-1:-1].float(),-1)
    values=lp.gather(-1,ids[n:].cuda()[:,None]).squeeze(-1)
    values=torch.cat([values[:-1],torch.logsumexp(lp[-1,stops],0)[None]])
    return float(values.mean()),lp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['qwen3','mistral']);args=ap.parse_args()
    family=args.family
    rows=[json.loads(x) for x in (ROUND/'compiled_v3'/f'{family}_train.jsonl').read_text().splitlines()]
    rows=sorted([x for x in rows if x.get('has_trajectory')],key=lambda x:x['id'])
    if family=='qwen3':
        base=str(_release_model)
        anchor=ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'
    else:
        base='/common_data/model/Mistral-7B-Instruct-v0.3'
        anchor=ROOT/'runs/aequal_kl_dev/k0_conservative/adapter'
    tok=AutoTokenizer.from_pretrained(base,use_fast=True);tok._equal_rag_disable_thinking=family=='qwen3';stops=eos_ids(tok)
    pairs=[compiled_pair(tok,r,stops) for r in rows]
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,str(anchor),adapter_name='baseline')
    for arm in ['sft','pair']:model.load_adapter(str(ROUND/f'{family}_{arm}'/'interp05/adapter'),adapter_name=arm,is_trainable=False)
    model=model.cuda();model.config.use_cache=False
    records=[];means={}
    with torch.inference_mode():
        for arm in ['baseline','sft','pair']:
            model.set_adapter(arm);model.eval();values=[]
            for row,(chosen,rejected,n,spec) in zip(rows,pairs):
                cp,lp=branch(model,chosen,n,stops)
                end=lp[-1];non=end.clone();non[stops]=-torch.inf
                greedy_end_margin=float(end[stops].max()-non.max())
                # Find the first *actual* divergent token, not a common closure
                # token that both branches must emit after the shared state.
                j=n
                while j<min(len(chosen),len(rejected)) and chosen[j]==rejected[j]:j+=1
                assert j<len(chosen)
                first=lp[j-n];target=int(chosen[j])
                rivals=first.clone()
                if target in stops:
                    rivals[stops]=-torch.inf;first_margin=float(first[stops].max()-rivals.max())
                else:
                    rivals[target]=-torch.inf;first_margin=float(first[target]-rivals.max())
                stop_probability=float(torch.logsumexp(end[stops],0).exp())
                del lp,end,non,first,rivals
                rp,rlp=branch(model,rejected,n,stops);del rlp
                record={'id':row['id'],'arm':arm,'action':spec['action'],'chosen_mean_logp':cp,
                        'rejected_mean_logp':rp,'preference_margin':cp-rp,
                        'first_divergent_target_greedy_margin':first_margin,
                        'completed_target_greedy_eos_margin':greedy_end_margin,
                        'completed_target_total_eos_probability':stop_probability}
                records.append(record);values.append(record)
            means[arm]={
                'n':len(values),'preference_margin':float(np.mean([x['preference_margin'] for x in values])),
                'margin1_satisfied':sum(x['preference_margin']>=1 for x in values),
                'first_divergent_target_greedy_wins':sum(x['first_divergent_target_greedy_margin']>0 for x in values),
                'completed_target_greedy_eos_wins':sum(x['completed_target_greedy_eos_margin']>0 for x in values),
                'total_eos_probability':float(np.mean([x['completed_target_total_eos_probability'] for x in values])),
                'by_training_action':{a:{'n':sum(x['action']==a for x in values),
                    'first_target_greedy_wins':sum(x['action']==a and x['first_divergent_target_greedy_margin']>0 for x in values)}
                    for a in ['stop','complete_then_stop']}}
            print(json.dumps({'family':family,'arm':arm,'diagnostic':means[arm]}),flush=True)
    out=ROUND/'mechanism_diagnostic';out.mkdir(exist_ok=True)
    target=out/f'{family}_summary.json'
    if target.exists():raise FileExistsError('Refusing overwrite')
    (out/f'{family}_states.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    report={'status':'COMPLETE','family':family,'means':means,'weights_updated':False,
            'heldout_inference':False,'new_free_generation':False,
            'warning':'In-sample teacher-forced repaired branches. Greedy token wins do not establish rollout success, independent quality, or efficiency. Max individual EOS, not summed class probability, determines a greedy stop.',
            'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
    target.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()

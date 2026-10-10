"""Bounded training-prefix replay to allow punctuation/answer-order variants."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import argparse
import hashlib
import json
from collections import Counter
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from completion_pairs_20261010 import ROOT,ROUND,compiled_pair,norm,units,covered
from train_verified_boundary_20261010 import eos_ids


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True,choices=['qwen3','mistral']);args=ap.parse_args();family=args.family
    rows=[json.loads(x) for x in (ROUND/'compiled_v3'/f'{family}_train.jsonl').read_text().splitlines()]
    active=sorted([x for x in rows if x.get('has_trajectory')],key=lambda x:hashlib.sha256(str(x['id']).encode()).hexdigest())
    stop=[x for x in active if x['completion_pair']['action']=='stop'][:8]
    rest=[x for x in active if x['completion_pair']['action']=='complete_then_stop'][:16-len(stop)]
    rows=stop+rest;assert len(rows)==16
    if family=='qwen3':
        base=str(_release_model);anchor=ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'
    else:
        base='/common_data/model/Mistral-7B-Instruct-v0.3';anchor=ROOT/'runs/aequal_kl_dev/k0_conservative/adapter'
    tok=AutoTokenizer.from_pretrained(base,use_fast=True);tok._equal_rag_disable_thinking=family=='qwen3';stops=eos_ids(tok)
    contexts=[]
    for row in rows:
        chosen,_,n,spec=compiled_pair(tok,row,stops)
        text=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,
                                    add_generation_prompt=True,enable_thinking=family!='qwen3')
        prompt=tok(text)['input_ids']
        assert chosen[:len(prompt)].tolist()==prompt,'Replay assistant-prefix mismatch'
        contexts.append((chosen[:n],len(prompt)))
    model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,str(anchor),adapter_name='baseline')
    for arm in ['sft','pair']:model.load_adapter(str(ROUND/f'{family}_{arm}'/'interp05/adapter'),adapter_name=arm,is_trainable=False)
    model=model.cuda();model.config.use_cache=True
    records=[];table={}
    with torch.inference_mode():
        for arm in ['baseline','sft','pair']:
            model.set_adapter(arm);model.eval();part=[]
            for index,(row,(ids,promptlen)) in enumerate(zip(rows,contexts)):
                output=model.generate(input_ids=ids[None].cuda(),attention_mask=torch.ones_like(ids)[None].cuda(),
                                      max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True)[0]
                new=output[len(ids):].tolist();text=tok.decode(output[promptlen:].tolist(),skip_special_tokens=True).strip()
                hit=covered(text,row['edges']);counts=Counter(norm(u) for _,_,u in units(text))
                repeated=sum(v-1 for v in counts.values() if v>1)
                ended=bool(new and new[-1] in stops)
                r={'id':row['id'],'arm':arm,'action':row['completion_pair']['action'],'output':text,
                   'new_token_ids':new,'new_tokens':len(new),'ended_by_eos':ended,
                   'hit_cap_without_eos':len(new)>=128 and not ended,'covered_edges':len(hit),
                   'required_edges':len(row['edges']),'repeated_units':repeated,
                   'local_complete_nonrepeat_eos':len(hit)==len(row['edges']) and not repeated and ended}
                records.append(r);part.append(r)
                if (index+1)%8==0:print(f'{family} replay {arm} {index+1}/16',flush=True)
            table[arm]={'n':len(part),'local_complete_nonrepeat_eos':sum(x['local_complete_nonrepeat_eos'] for x in part),
                        'mean_new_tokens':sum(x['new_tokens'] for x in part)/len(part),
                        'cap_hits':sum(x['hit_cap_without_eos'] for x in part),
                        'repetition_cases':sum(x['repeated_units']>0 for x in part),
                        'all_required_edges_covered':sum(x['covered_edges']==x['required_edges'] for x in part),
                        'by_action':{a:{'n':sum(x['action']==a for x in part),
                            'local_complete_nonrepeat_eos':sum(x['action']==a and x['local_complete_nonrepeat_eos'] for x in part)}
                            for a in ['stop','complete_then_stop']}}
    folder=ROUND/'prefix_replay';folder.mkdir(exist_ok=True);target=folder/f'{family}_summary.json'
    if target.exists():raise FileExistsError('Refusing overwrite')
    (folder/f'{family}_predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
    report={'status':'COMPLETE','family':family,'table':table,'selected_training_ids':[x['id'] for x in rows],
            'heldout_predictions':0,'max_new_tokens':128,'weights_updated':False,
            'warning':'16 hash-selected training states, stop-stratified 8 where available. Retokenized shared prefixes, not recovery of original generated IDs. Success checks constructed lexical edges, permits punctuation/order variants, and cannot retract prior prefix errors. No independent efficacy or entailment claim.'}
    target.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

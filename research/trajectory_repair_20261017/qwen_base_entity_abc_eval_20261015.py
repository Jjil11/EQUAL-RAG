"""Gold-free first-token blocking probe and matched full-prompt decoding."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import time
import qwen_equal_graph_20261014 as q
from qwen_base_entity_abc_data_20261015 import RUN,STOPS
from entity_abc_contracts_20261015 import parse_closed,blocked_first_ids,inventory,normalize


def load(which,train=False):
    import torch
    from peft import LoraConfig,get_peft_model,PeftModel
    model,tok,mapping=q.load_model('base')
    if which=='anchor':
        model=PeftModel.from_pretrained(model,str(q.ANCHOR),is_trainable=False)
    elif which in ['base_ce','base_pair']:
        if train:
            q.seed_all(20261015)
            model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
            model.gradient_checkpointing_enable();model.enable_input_require_grads()
        else:model=PeftModel.from_pretrained(model,str(RUN/which/'adapter'),is_trainable=False)
    elif which not in ['base','base_block']:raise ValueError(which)
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.config.use_cache=not train
    return model,tok,mapping


class EntityFirstTokenBlocker:
    """Raw requested probe: observed names only, no gold/alias lists at decode."""
    def __init__(self,tok,start):self.tok=tok;self.start=start;self.events=[];self.names={}
    def __call__(self,input_ids,scores):
        import re
        ids=input_ids[0,self.start:].tolist();text=self.tok.decode(ids,skip_special_tokens=True)
        if not re.search(r'\]\s*,\s*$',text):return scores
        # Parse closed prefix; do not use any supported-entity inventory here.
        closed=text.rstrip().rstrip(',').rstrip();parsed=parse_closed(closed)
        if parsed is None or parsed[1].strip() not in ['', '.']:return scores
        names=sorted({u['name'] for u in parsed[0]})
        blocked=blocked_first_ids(self.tok,names)
        if not blocked:return scores
        top=int(scores[0].argmax());self.events.append({'generated_position':len(ids),
            'blocked_ids':blocked,'observed_names':names,'original_top1':top,'top1_was_blocked':top in blocked})
        scores[0,blocked]=-float('inf')
        return scores


def generate(which,collection=False):
    import torch
    from transformers import LogitsProcessorList
    target=RUN/('collection' if collection else 'eval')/which
    target.mkdir(parents=True,exist_ok=False);q.dump(target/'status.json',{'state':'LOADING'})
    try:
        model,tok,mapping=load(which);model.eval()
        rows=q.read(RUN/('collection_inputs128.jsonl' if collection else 'eval69.jsonl'))
        q.dump(target/'manifest.json',{'model':which,'official_base_initialization':True,
            'adapter_sha256':q.sha(RUN/which/'adapter/adapter_model.safetensors') if which in ['base_ce','base_pair'] else None,
            'input_sha256':q.sha(RUN/('collection_inputs128.jsonl' if collection else 'eval69.jsonl')),
            'placement':mapping,'greedy':True,'max_new_tokens':256,'no_truncation':True,
            'probe_uses_gold':False,'script_sha256':q.sha(__file__)})
        start=time.time()
        with (target/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode():
            for i,row in enumerate(rows):
                prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                enc=tok(prompt,return_tensors='pt').to(model.get_input_embeddings().weight.device)
                n=enc['input_ids'].shape[1];assert n<=2048
                blocker=EntityFirstTokenBlocker(tok,n)
                extra={'logits_processor':LogitsProcessorList([blocker])} if which=='base_block' else {}
                started=time.time()
                ids=model.generate(**enc,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id,**extra)[0,n:].tolist()
                p={'id':row['id'],'eval_split':row.get('eval_split'),'tokens':len(ids),
                    'output':tok.decode(ids,skip_special_tokens=True).strip(),
                    'prompt_token_ids':enc['input_ids'][0].tolist(),'generated_token_ids':ids,
                    'ended_by_eos':bool(ids and ids[-1] in STOPS),
                    'hit_cap_without_eos':bool(len(ids)==256 and ids[-1] not in STOPS),
                    'generation_seconds':time.time()-started,'blocking_events':blocker.events}
                # Offline audit only, AFTER generation; never feeds decoder scores.
                if which=='base_block':
                    groups=inventory(row);collisions=[]
                    for event in blocker.events:
                        seen={normalize(name) for name in event['observed_names']}
                        for group in groups:
                            if not group['supporting_doc_ids'] or seen&set(group['values']):continue
                            for name in group['aliases']:
                                first=tok(' '+name,add_special_tokens=False)['input_ids'][0]
                                if first in event['blocked_ids']:collisions.append({'position':event['generated_position'],
                                    'unseen_known_lexical_name':name,'shared_first_token':first})
                    p['offline_unseen_supported_first_token_collisions']=collisions
                f.write(json.dumps(p,ensure_ascii=False)+'\n');f.flush()
                q.dump(target/'status.json',{'state':'GENERATING','completed':i+1,'of':len(rows),'seconds':time.time()-start})
                if (i+1)%8==0:print(json.dumps({'model':which,'completed':i+1,'of':len(rows),'seconds':time.time()-start}),flush=True)
        q.dump(target/'status.json',{'state':'COMPLETE','rows':len(rows),'seconds':time.time()-start})
    except Exception:q.dump(target/'status.json',{'state':'FAILED'});raise


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--collection',action='store_true');a=ap.parse_args()
    generate(a.model,a.collection)

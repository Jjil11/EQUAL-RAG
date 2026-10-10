"""On-policy top5 mining plus exact-format frozen-anchor cache, train IDs only."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import argparse
import hashlib
import json
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
import train_trajectory_equal as legacy
from build_qwen_retrieval_align_20261010 import RUN,CACHE
from completion_pairs_20261010 import true_repeat,retention_mask,encode_end,covered
from qwen_clean_labels_20261010 import compiled_pair,candidates
from qwen_priority_compile_20261010 import earliest_invalid


def text_hash(s):return hashlib.sha256(s.encode()).hexdigest()


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);args=ap.parse_args()
    assert args.shard in range(4)
    target=RUN/'mining'/f'shard_{args.shard}.jsonl'
    temporary=target.with_suffix('.jsonl.inprogress')
    if target.exists() or temporary.exists():raise FileExistsError('Refusing duplicate/resume without audit')
    data=RUN/'data/train_top5.jsonl';manifest=json.loads((RUN/'data/manifest.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest()==manifest['data_sha256']
    rows=[json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    selected=[(i,r) for i,r in enumerate(rows) if i%4==args.shard]
    base=str(_release_model)
    adapter=str(_release_root / 'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter')
    tok=AutoTokenizer.from_pretrained(base,use_fast=True);tok._equal_rag_disable_thinking=True
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    stops=GenerationConfig.from_pretrained(base).eos_token_id
    stops=[stops] if isinstance(stops,int) else stops
    model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16),adapter).cuda().eval()
    count=active=cap_hits=uncertain_repeats=0;started=time.time()
    with temporary.open('w') as handle:
        for index,row in selected:
            prefix=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,
                                          add_generation_prompt=True,enable_thinking=False)
            enc=tok(prefix,return_tensors='pt',truncation=False).to('cuda')
            assert enc['input_ids'].shape[1]+260<=2048
            model.config.use_cache=True
            out=model.generate(**enc,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)
            ids=out[0,enc['input_ids'].shape[1]:].tolist()
            row['short']=tok.decode(ids,skip_special_tokens=True).strip()
            row['generation_token_ids']=ids;row['anchor_output_tokens']=len(ids)
            row['anchor_ended_by_eos']=bool(ids and ids[-1] in stops)
            row['anchor_cap_hit']=len(ids)>=256 and not row['anchor_ended_by_eos']
            repeated=true_repeat(row);row['verified_surface_repeat']=bool(repeated)
            row['has_trajectory']=bool(repeated and row['edges'])
            if row['has_trajectory']:
                prefix,unit=repeated;row['trajectory_prefix']=prefix;row['trajectory_continuation']=', '+unit
                _,rejected,n,spec=compiled_pair(tok,row,stops)
                texts=candidates(row)
                assert all(len(covered(t,row['edges']))==len(row['edges']) and not true_repeat({'short':t}) for t in texts)
                options=[encode_end(tok,row['prompt'],t,stops)[0].tolist() for t in texts]
                assert all(len(x)<=2048 for x in options)
                row['completion_pair']=spec;row['action_decision']=earliest_invalid(rejected.tolist(),options,n,set(stops))
                row['candidate_count']=len(texts);active+=1
            elif repeated:uncertain_repeats+=1
            teacher,_,_=encode_end(tok,row['prompt'],row['short'],stops)
            original,_,labels=legacy.encode_example(tok,row['prompt'],row['short'],2048)
            assert len(teacher)<=2048 and original[:len(teacher)].tolist()==teacher.tolist()
            positions=(labels[1:]!=-100).nonzero().flatten().tolist()
            full=[p+1<len(teacher) for p in positions]
            row['full_retention_mask']=full
            row['failure_local_retention_mask']=retention_mask(tok,row).tolist() if row['has_trajectory'] else full
            model.config.use_cache=False
            _,_,_,logits=legacy.response_logp(model,tok,row['prompt'],row['short'],2048,True)
            assert len(logits)==len(full)==len(row['failure_local_retention_mask'])
            cache=legacy.anchor_cache_path(str(CACHE),row['id'])
            if cache.exists():raise FileExistsError('Cache collision')
            tensor=logits.to(device='cpu',dtype=torch.bfloat16).contiguous()
            torch.save(tensor,cache)
            row['cache_metadata']={'path':str(cache),'shape':list(tensor.shape),'dtype':'bfloat16',
                'prompt_sha256':text_hash(row['prompt']),'response_sha256':text_hash(row['short']),
                'warning':'KL uses chat-template re-tokenized decoded response, not original generation IDs.'}
            del out,enc,logits,tensor
            row['global_index']=index;handle.write(json.dumps(row,ensure_ascii=False)+'\n');handle.flush()
            count+=1;cap_hits+=int(row['anchor_cap_hit'])
            if count%10==0 or count==len(selected):print(json.dumps({'shard':args.shard,'done':count,
                'total':len(selected),'active':active,'cap_hits':cap_hits,'uncertain_repeats':uncertain_repeats,
                'elapsed_seconds':time.time()-started}),flush=True)
    temporary.rename(target)
    (RUN/'mining'/f'shard_{args.shard}_manifest.json').write_text(json.dumps({'shard':args.shard,'rows':count,
        'active':active,'cap_hits':cap_hits,'uncertain_repeats':uncertain_repeats,'eos_ids':stops,
        'data_sha256':manifest['data_sha256'],'shard_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()

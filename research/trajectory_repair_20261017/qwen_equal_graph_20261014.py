"""Evidence-equivalent self edits, Qwen-first bounded exploration.

Observed response equivalence is not a certificate of factual correctness or
exhaustive evidence coverage. No sealed inputs are read by this runner.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

ROOT = Path(str(_release_root))
RUN = ROOT / 'runs/qwen_equal_graph_20261014'
SOURCE = ROOT / 'runs/qwen_signal_split_20261014'
MODEL = str(_release_model)
ANCHOR = ROOT / 'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    with Path(path).open('x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')


def cited_units(text):
    """Conservative citation-closed list parser; internal name commas survive.

    Reject free-form introductions, malformed separators and uncited tails.
    Different citation sets are different units and are NEVER collapsed.
    """
    result=[]
    pos=0
    pat=re.compile(r'(.+?)(\[\d+\](?:\s*\[\d+\])*)')
    for m in pat.finditer(text):
        if m.start()!=pos:
            return None
        name=m.group(1)
        if result:
            if not re.match(r'^\s*,\s*',name):
                return None
            name=re.sub(r'^\s*,\s*','',name,count=1)
        name=name.strip()
        if not name or '[' in name or ']' in name or '\n' in name:
            return None
        refs=tuple(sorted(set(int(x) for x in re.findall(r'\[(\d+)\]',m.group(2)))))
        surface=name+' '+m.group(2).strip()
        key=(re.sub(r'\s+',' ',name).casefold(),refs)
        result.append({'name':name,'refs':refs,'surface':surface,'key':key})
        pos=m.end()
    if not result or text[pos:].strip() not in ('','.'):return None
    return result


def compact_equivalent(text):
    atoms=cited_units(text)
    if atoms is None:return None
    seen=set();kept=[]
    for a in atoms:
        if a['key'] not in seen:
            kept.append(a);seen.add(a['key'])
    candidate=', '.join(a['surface'] for a in kept)
    if text.rstrip().endswith('.'):candidate+='.'
    parsed=cited_units(candidate)
    assert parsed is not None and {a['key'] for a in parsed}=={a['key'] for a in atoms}
    return {'chosen':candidate,'rejected':text,'unique_units':len(kept),
            'removed_duplicates':len(atoms)-len(kept),'units':kept,
            'equivalence':'Exact observed name plus citation-set inventory; NOT semantic closure.'}


def compact_censored(text):
    """Keep an uncited cap-truncated suffix verbatim; never infer its meaning.

    Only already closed duplicated units are removed. This extends strict
    parsing without silently discarding possibly new unfinished content.
    """
    strict=compact_equivalent(text)
    if strict is not None:return dict(strict,censored_tail='')
    citations=list(re.finditer(r'\[\d+\](?:\s*\[\d+\])*',text))
    if not citations:return None
    end=citations[-1].end();tail=text[end:]
    if not re.match(r'^\s*,',tail):return None
    prefix=compact_equivalent(text[:end])
    if prefix is None:return None
    return dict(prefix,chosen=prefix['chosen']+tail,rejected=text,censored_tail=tail)


def seed_all(seed=20261014):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def load_model(which,train=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    from accelerate import dispatch_model
    seed_all()
    assert torch.cuda.device_count()==2
    free=[torch.cuda.mem_get_info(i)[0]/2**30 for i in range(2)]
    if min(free)<21:raise RuntimeError(f'Insufficient free GPU memory; no foreign process changed: {free}')
    # Two-GPU split stays below the observed free memory, without CPU offload.
    for i in range(2):torch.cuda.set_per_process_memory_fraction(.49,i)
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
    tok._equal_rag_disable_thinking=True
    base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    if which=='base':
        assert not train
        model=base;prefix='model';head='lm_head'
    else:
        folder=ANCHOR if which=='anchor' else RUN/which/'adapter'
        model=PeftModel.from_pretrained(base,str(folder),is_trainable=train)
        prefix='base_model.model.model';head='base_model.model.lm_head'
    mapping={prefix+'.embed_tokens':0,prefix+'.norm':1,head:1}
    mapping.update({prefix+f'.layers.{i}':0 if i<20 else 1 for i in range(40)})
    model=dispatch_model(model,device_map=mapping)
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
    if train:
        model.gradient_checkpointing_enable();model.enable_input_require_grads()
    model.config.use_cache=not train
    return model,tok,mapping


def prepare():
    assert not RUN.exists(),'Never overwrite an existing round'
    rows=read(SOURCE/'train.jsonl');val=read(SOURCE/'validation.jsonl')
    assert len(rows)==128 and len(val)==48
    assert not {r['question_id'] for r in rows}&{r['question_id'] for r in val}
    RUN.mkdir()
    write_rows(RUN/'train_inputs.jsonl',rows);write_rows(RUN/'validation.jsonl',val)
    dump(RUN/'protocol.json',{
        'model':'Qwen3-14B','starting_adapter':str(ANCHOR),'starting_adapter_sha256':sha(ANCHOR/'adapter_model.safetensors'),
        'train_input_sha256':sha(RUN/'train_inputs.jsonl'),'validation_sha256':sha(RUN/'validation.jsonl'),
        'seed':20261014,'arms':['self_sft','graph_preference'],'lr':2e-6,'epochs':1,'grad_accum':8,
        'training_limit':128,'max_new_tokens':256,'max_prompt_tokens':2048,'thinking':False,'decoding':'greedy',
        'content_equivalence':'Preserve ALL distinct observed name-citation-set units, including unverified units; never first-repeat truncation.',
        'closure_policy':'Native teacher EOS is behavioral self-distillation, not exhaustive closure. Capped outputs do not receive hard EOS labels.',
        'eos_ce_weight':.1,'body_ce_weight':.1,'preference_weight':.1,'beta':.1,
        'reference':'Frozen original pref0.4 adapter; cached response-average log probabilities for identical token IDs.',
        'counterfactuals':'Full repeated original; delete one known supported observed unit; change its citation to a non-supporting visible document.',
        'quality_label_scope':'Known lexical/local citation edges only; does not certify relation entailment.',
        'fairness':'Both arms same inputs, chosen labels, order, forward/backward branches, update count; preference gradient zero in self_sft. Gradient magnitude differs.',
        'evaluation':'48 already reused development inputs, two views of 24 questions. Exploratory only, no sealed or independent confirmation.',
        'point_screen':'Each view Q/F1 >= original anchor-.02, tokens <= .9*original anchor; also Q/F1 >= anchor+first-repeat-.02 and tokens <= anchor+first-repeat. Original-base comparison reported separately.',
        'no_automatic_expansion':True,'sealed_predictions':0,'human_annotations':0,'extra_models':0,
        'purpose':'Return to evidence-equivalent response-level learning; not a promise of novelty or improvement.',
        'script_sha256':sha(__file__)})


def generate(which,phase,formatted=False):
    import torch
    output_root=ROOT/'runs/qwen_equal_graph_format_20261014' if formatted else RUN
    assert not formatted or phase=='eval'
    out=output_root/('collection' if phase=='collect' else 'eval')/which
    out.mkdir(parents=True,exist_ok=False);dump(out/'status.json',{'state':'LOADING'})
    try:
        model,tok,mapping=load_model(which)
        rows=read(output_root/('train_inputs.jsonl' if phase=='collect' else 'validation.jsonl'))
        model.eval();stops=model.generation_config.eos_token_id
        stops=[stops] if isinstance(stops,int) else list(stops)
        manifest={'model':which,'base_only':which=='base','adapter_sha256':None if which=='base' else sha((ANCHOR if which=='anchor' else RUN/which/'adapter')/'adapter_model.safetensors'),
                  'prompt_sha256':sha(output_root/('train_inputs.jsonl' if phase=='collect' else 'validation.jsonl')),
                  'rows':len(rows),'stops':stops,'placement':mapping,'precision':'BF16','code_sha256':sha(__file__)}
        dump(out/'manifest.json',manifest);began=time.time();records=[]
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode():
            for i,r in enumerate(rows):
                text=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                enc=tok(text,return_tensors='pt').to(model.get_input_embeddings().weight.device)
                assert len(enc['input_ids'][0])<=2048,'Prompt truncation prohibited'
                ids=model.generate(**enc,max_new_tokens=256,do_sample=False,pad_token_id=tok.eos_token_id)[0,len(enc['input_ids'][0]):].tolist()
                record={'id':r['id'],'question_id':r['question_id'],'view':r['view'],'prompt_ids':enc['input_ids'][0].tolist(),
                        'generated_ids':ids,'tokens':len(ids),'output':tok.decode(ids,skip_special_tokens=True).strip(),
                        'ended_by_eos':bool(ids and ids[-1] in stops),'hit_cap_without_eos':len(ids)==256 and ids[-1] not in stops,'eos_ids':stops}
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();records.append(record)
                progress={'state':'GENERATING','phase':phase,'model':which,'completed':i+1,'of':len(rows),'seconds':time.time()-began}
                dump(out/'status.json',progress)
                if (i+1)%8==0:print(json.dumps(progress),flush=True)
        if phase=='eval':score_records(rows,records,out)
        dump(out/'status.json',{'state':'COMPLETE','rows':len(rows),'seconds':time.time()-began})
    except Exception:
        dump(out/'status.json',{'state':'FAILED'});raise


def score_records(rows,records,out):
    from score_strict_units_20261010 import score,summarize
    scores=[]
    for r,p in zip(rows,records):
        assert r['id']==p['id']
        pp=dict(p,grounded_edges=0)
        scores.append(score(r,pp))
    write_rows(out/'scored.jsonl',scores)
    tables={}
    for view in ['raw','rich']:
        ids={r['id'] for r in rows if r['view']==view}
        tables[view]=dict(summarize([r for r in scores if r['id'] in ids]),
            true_cap_hit_rate=sum(p['hit_cap_without_eos'] for p in records if p['id'] in ids)/len(ids))
    dump(out/'summary.json',tables)
    print(json.dumps({'summaries':tables},ensure_ascii=False),flush=True)


def compile_pairs():
    from transformers import AutoTokenizer
    from completion_pairs_20261010 import covered,edge_matches
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True);tok._equal_rag_disable_thinking=True
    source={r['id']:r for r in read(RUN/'train_inputs.jsonl')}
    pairs=[];skipped=[];contrast_skips=[]
    for p in read(RUN/'collection/anchor/predictions.jsonl'):
        row=source[p['id']]
        edit=compact_equivalent(p['output']) if p['ended_by_eos'] else compact_censored(p['output'])
        if edit is None:
            skipped.append({'id':p['id'],'reason':'non_exhaustive_list_parse'});continue
        n_chosen=len(tok(edit['chosen'],add_special_tokens=False)['input_ids'])
        n_rejected=len(tok(p['output'],add_special_tokens=False)['input_ids'])
        # Identity cases remain supervision anchors and are NOT efficiency pairs.
        pair=dict(id=p['id'],question_id=row['question_id'],view=row['view'],prompt=row['prompt'],
                  chosen=edit['chosen'],original=p['output'],terminal_observed=p['ended_by_eos'],
                  chosen_tokens=n_chosen,original_tokens=n_rejected,removed_duplicates=edit['removed_duplicates'],
                  censored_tail=edit.get('censored_tail',''),
                  inventory=[{'name':u['name'],'refs':list(u['refs'])} for u in edit['units']],negatives=[])
        assert covered(pair['chosen'],row['edges'])==covered(p['output'],row['edges'])
        if edit['removed_duplicates'] and n_chosen<n_rejected:
            pair['negatives'].append({'kind':'redundant','text':p['output']})
        atoms=edit['units'];known=[i for i,u in enumerate(atoms) if edge_matches(u['surface'],row['edges'])]
        if known and len(atoms)>1:
            selected=known[int(hashlib.sha256(p['id'].encode()).hexdigest()[:8],16)%len(known)]
            drop=', '.join(u['surface'] for i,u in enumerate(atoms) if i!=selected)+pair['censored_tail']
            if not pair['censored_tail'] and pair['chosen'].endswith('.'):drop+='.'
            if len(covered(drop,row['edges']))<len(covered(pair['chosen'],row['edges'])):
                pair['negatives'].append({'kind':'drop','text':drop})
            else:
                contrast_skips.append({'id':p['id'],'kind':'drop','reason':'other_observed_unit_retains_same_known_coverage'})
            matches=edge_matches(atoms[selected]['surface'],row['edges'])
            support=set().union(*(set(row['edges'][i].get('supporting_doc_ids',[row['edges'][i]['doc_id']])) for i in matches))
            alternatives=[d for d in range(1,len(row['docs'])+1) if d not in support]
            if alternatives:
                swap=[u['surface'] for u in atoms]
                swap[selected]=atoms[selected]['name']+f' [{alternatives[0]}]'
                sw=', '.join(swap)+pair['censored_tail']+('.' if not pair['censored_tail'] and pair['chosen'].endswith('.') else '')
                if len(covered(sw,row['edges']))<len(covered(pair['chosen'],row['edges'])):
                    pair['negatives'].append({'kind':'swap','text':sw})
                else:
                    contrast_skips.append({'id':p['id'],'kind':'swap','reason':'other_observed_unit_retains_same_known_coverage'})
        pairs.append(pair)
    out=RUN/'compiled';out.mkdir()
    write_rows(out/'pairs.jsonl',pairs)
    counts={k:sum(any(n['kind']==k for n in p['negatives']) for p in pairs) for k in ['redundant','drop','swap']}
    report={'rows':len(pairs),'questions':len({p['question_id'] for p in pairs}),'counts':counts,
            'natural_eos_rows':sum(p['terminal_observed'] for p in pairs),
            'censored_rows':sum(not p['terminal_observed'] for p in pairs),
            'censored_tail_preserved_rows':sum(bool(p['censored_tail']) for p in pairs),
            'all_observed_distinct_units_preserved':True,'all_known_coverage_preserved':True,
            'skipped':skipped,'contrast_skips':contrast_skips,'pairs_sha256':sha(out/'pairs.jsonl'),
            'gate':len(pairs)>=64 and counts['redundant']>=16 and counts['drop']>=16,
            'closure_warning':'Semantic/exhaustive closure UNKNOWN; natural EOS is only a teacher behavioral prior.'}
    dump(out/'audit.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--compile',action='store_true')
    ap.add_argument('--generate',choices=['base','anchor','self_sft','graph_preference']);ap.add_argument('--phase',choices=['collect','eval'],default='eval')
    ap.add_argument('--formatted',action='store_true');args=ap.parse_args()
    if args.prepare:prepare()
    elif args.compile:compile_pairs()
    else:generate(args.generate,args.phase,args.formatted)


if __name__=='__main__':main()

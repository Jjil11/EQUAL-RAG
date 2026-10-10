"""Use recorded on-policy token prefixes only when existing evidence is covered."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from compile_qwen_terminal_bridge_20261010 import ROOT,RUN as SOURCE,CACHE
from completion_pairs_20261010 import units,norm,covered,true_repeat
RUN=ROOT/'runs/qwen_native_terminal_20261010'

def main():
    if RUN.exists():raise FileExistsError('Refusing round overwrite')
    src=SOURCE/'compiled/train.jsonl';sha=hashlib.sha256(src.read_bytes()).hexdigest()
    assert sha=='2c0515fb47b2f6e29babfe3c91582815e0695919b462fec8897ee4e4a3be2953'
    rows=[json.loads(s) for s in src.read_text().splitlines() if s.strip()]
    tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
    changed=[]
    for r in rows:
        if not r['has_trajectory']:continue
        r['terminal_decision']['origin']='verified_completed_counterfactual'
        if r['completion_pair']['action']!='stop':continue
        seen=set();boundary=None
        for lo,hi,u in units(r['short']):
            v=norm(u)
            if v in seen:boundary=lo;break
            seen.add(v)
        assert boundary is not None
        text=r['short'][:boundary].rstrip();ids=r['generation_token_ids']
        if len(covered(text,r['edges']))!=len(r['edges']) or true_repeat({'short':text}):continue
        assert tok.decode(ids,skip_special_tokens=True).strip()==r['short']
        matches=[k for k in range(1,len(ids)+1) if tok.decode(ids[:k],skip_special_tokens=True).strip()==text]
        if not matches:continue
        k=max(matches);prompt=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        context=prompt+ids[:k];old=r['terminal_decision']['context_ids'];assert context!=old and len(context)<2048
        r['terminal_decision'].update(context_ids=context,origin='recorded_onpolicy_prefix',generation_prefix_tokens=k)
        changed.append(r['id'])
    assert len(rows)==350 and len(changed)==72 and sum(r['has_trajectory'] for r in rows)==150
    out=RUN/'compiled';out.mkdir(parents=True);p=out/'train.jsonl'
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    active=[r for r in rows if r['has_trajectory']]
    smoke=[r for r in active if r['terminal_decision']['origin']=='recorded_onpolicy_prefix'][:4]+[r for r in active if r['completion_pair']['action']=='complete_then_stop'][:4]
    (out/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke))
    audit={'rows':350,'active':150,'explicit_eos_sites':150,'native_eos_sites':72,'counterfactual_eos_sites':78,
        'stop':75,'completion':75,'zero_lexical_support':106,'smoke_rows':8,'source_sha256':sha,
        'compiled_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changed_ids':changed,'development_or_holdout_inputs_read':0,
        'warning':'Only lexical-coverage-certified stop cases use native prefixes; unknown/citation-incomplete cases are not forced to stop.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()

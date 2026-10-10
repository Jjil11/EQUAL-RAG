"""Train-only explicit EOS site after the already verified completed repair."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer,GenerationConfig
from completion_pairs_20261010 import ROOT,encode_end,covered,true_repeat
SOURCE=ROOT/'runs/qwen_retrieval_align_20261010'
RUN=ROOT/'runs/qwen_terminal_bridge_20261010'
CACHE=Path(str(_release_cache / 'qwen_retrieval_align_20261010'))

def main():
    if RUN.exists():raise FileExistsError('Refusing new round overwrite')
    src=SOURCE/'compiled/train.jsonl';sha=hashlib.sha256(src.read_bytes()).hexdigest()
    assert sha=='3bd4f89a86a91f5bfd60f94f730edbfbe4337a3c163749b57f3f4f26956945ef'
    tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
    tok._equal_rag_disable_thinking=True
    stops=GenerationConfig.from_pretrained(tok.name_or_path).eos_token_id
    stops=[stops] if isinstance(stops,int) else stops
    rows=[json.loads(s) for s in src.read_text().splitlines() if s.strip()];active=[]
    for r in rows:
        if not r['has_trajectory']:continue
        chosen=r['completion_pair']['chosen']
        assert not true_repeat({'short':chosen}) and len(covered(chosen,r['edges']))==len(r['edges'])
        ids,_,_=encode_end(tok,r['prompt'],chosen,stops)
        assert int(ids[-1]) in stops and len(ids)<=2048 and int(ids[-2]) not in stops
        r['terminal_decision']={'context_ids':ids[:-1].tolist(),'allowed_ids':stops,'canonical_eos_id':int(ids[-1])}
        active.append(r)
    assert len(rows)==350 and len(active)==150
    assert sum(bool(set(stops).intersection(r['action_decision']['allowed_ids'])) for r in active)==0
    out=RUN/'compiled';out.mkdir(parents=True)
    p=out/'train.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    selected=[r for r in active if r['completion_pair']['action']=='stop'][:4]+[r for r in active if r['completion_pair']['action']=='complete_then_stop'][:4]
    (out/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    audit={'rows':350,'active':150,'explicit_eos_sites':150,'earliest_sites_allowing_eos':0,'smoke_rows':8,
        'source_sha256':sha,'compiled_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'stop':75,'completion':75,'zero_lexical_support':106,'cache_reused_without_modification':str(CACHE),
        'development_or_holdout_inputs_read':0,'eos_ids':stops,
        'warning':'EOS contexts are teacher-forced completed repairs, not original on-policy terminal states; no full intermediate suffix CE.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()

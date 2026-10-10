"""Train-only noncontradictory completion labels; original files stay frozen."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
import re
from pathlib import Path
from completion_pairs_20261010 import ROOT,norm,units,covered,true_repeat,encode_end
from qwen_priority_compile_20261010 import earliest_invalid

RUN=ROOT/'runs/qwen_clean_labels_20261010'


def annotated_prefix(text,edges):
    """Correct known entity citation targets without emitting its name twice.

    Uses supplied lexical training annotation, not semantic entailment. Unknown
    entities in the old prefix cannot be certified or retracted by this rule.
    """
    replacements=[]
    for lo,hi,unit in units(text):
        matched=[e for e in edges if norm(unit) in {norm(a) for a in e['aliases']}]
        if not matched:continue
        refs={int(x) for x in re.findall(r'\[(\d+)\]',unit)}
        needed=[e for e in matched if not refs.intersection(e.get('supporting_doc_ids',[e['doc_id']]))]
        if not needed:continue
        union=set().union(*(set(e.get('supporting_doc_ids',[e['doc_id']])) for e in matched))
        kept=refs.intersection(union)
        for e in needed:kept.add(e['doc_id'])
        name=re.sub(r'\[\d+\]','',unit).strip()
        replacements.append((lo,hi,name+' '+''.join(f'[{d}]' for d in sorted(kept))))
    for lo,hi,value in reversed(replacements):text=text[:lo]+value+text[hi:]
    return text


def complete(prefix,edges,first=None,alias=None,doc=None):
    text=annotated_prefix(prefix,edges)
    if first is not None:
        e=edges[first];text=annotated_prefix(text+', '+f"{alias or e['answer']} [{doc or e['doc_id']}]",edges)
    while len(covered(text,edges))<len(edges):
        before=covered(text,edges);i=next(i for i in range(len(edges)) if i not in before)
        e=edges[i];candidate=text+', '+f"{e['answer']} [{e['doc_id']}]"
        text=annotated_prefix(candidate,edges)
        assert len(covered(text,edges))>len(before)
    if true_repeat({'short':text}):raise ValueError('Positive completion still repeats')
    return text


def repair(row):
    original=row['trajectory_prefix'].rstrip().rstrip(',').rstrip()
    prefix=annotated_prefix(original,row['edges']);seen=covered(prefix,row['edges'])
    missing=[i for i in range(len(row['edges'])) if i not in seen]
    chosen=complete(prefix,row['edges'])
    rejected=row['trajectory_prefix']+row['trajectory_continuation']
    assert chosen!=rejected
    return {'chosen':chosen,'rejected':rejected,'prefix':prefix,'missing_edges':missing,
            'action':'complete_then_stop' if missing else 'stop'}


def compiled_pair(tok,row,stops,max_length=2048):
    spec=repair(row)
    chosen,offsets,start=encode_end(tok,row['prompt'],spec['chosen'],stops)
    rejected,_,_=encode_end(tok,row['prompt'],spec['rejected'],stops)
    boundary=start+len(spec['prefix'])
    limit=next((i for i,(_,hi) in enumerate(offsets) if hi>boundary),len(chosen)-1)
    n=0
    for a,b in zip(chosen.tolist(),rejected.tolist()):
        if a!=b or n>=limit:break
        n+=1
    assert 0<n<min(len(chosen),len(rejected)) and max(len(chosen),len(rejected))<=max_length
    spec.update(shared_tokens=n,chosen_tokens=len(chosen)-n,rejected_tokens=len(rejected)-n,
                first_chosen_token=int(chosen[n]),first_rejected_token=int(rejected[n]))
    return chosen,rejected,n,spec


def candidates(row):
    spec=repair(row);prefix=spec['prefix'];missing=spec['missing_edges'];values=set()
    if not missing:return [prefix,prefix+'.']
    for i in missing:
        e=row['edges'][i];aliases=list(dict.fromkeys([e['answer']]+sorted(set(e['aliases']))))
        aliases=[a.strip() for a in aliases if norm(a) and '\n' not in a and '[' not in a and ']' not in a][:8]
        for alias in aliases:
            for doc in sorted(set(e.get('supporting_doc_ids',[e['doc_id']]))):
                text=complete(prefix,row['edges'],i,alias,doc)
                values.update([text,text+'.'])
    return sorted(values)


def main():
    from transformers import AutoTokenizer,GenerationConfig
    source=ROOT/'runs/qwen_priority_20261010/compiled/train.jsonl'
    expected='658c9f2d7aa9f8c1a820e470ecdaf697d7db9ae7e07888973a767ae327559a8f'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
    tok._equal_rag_disable_thinking=True
    stops=GenerationConfig.from_pretrained(tok.name_or_path).eos_token_id
    stops=[stops] if isinstance(stops,int) else stops
    audit=[]
    for row in rows:
        if not row.get('has_trajectory'):continue
        old=row['completion_pair'];chosen,rejected,n,spec=compiled_pair(tok,row,stops)
        texts=candidates(row);ids=[encode_end(tok,row['prompt'],text,stops)[0].tolist() for text in texts]
        assert all(len(c)<=2048 for c in ids)
        assert all(len(covered(text,row['edges']))==len(row['edges']) and not true_repeat({'short':text}) for text in texts)
        decision=earliest_invalid(rejected.tolist(),ids,n,set(stops))
        row['completion_pair']=spec;row['action_decision']=decision
        audit.append({'id':row['id'],'positive_changed':old['chosen']!=spec['chosen'],
            'prefix_changed':old['prefix']!=spec['prefix'],'old_positive_repeated':bool(true_repeat({'short':old['chosen']})),
            'new_positive_repeated':bool(true_repeat({'short':spec['chosen']})),
            'new_action':spec['action'],'old_shared':old['shared_tokens'],'new_shared':n,
            'candidate_count':len(texts)})
    assert len(rows)==350 and len(audit)==39 and sum(r['old_positive_repeated'] for r in audit)==7
    assert sum(r['new_positive_repeated'] for r in audit)==0
    out=RUN/'compiled';out.mkdir(parents=True,exist_ok=True);target=out/'train.jsonl'
    if target.exists():raise FileExistsError('Frozen clean labels exist')
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    active=[r for r in rows if r.get('has_trajectory')]
    chosen=[r for r in active if r['completion_pair']['action']=='stop'][:4]
    chosen += [r for r in active if r['completion_pair']['action']!='stop'][:8-len(chosen)]
    (out/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    report={'rows':350,'active':39,'source_sha256':expected,'compiled_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'old_positive_repeat_rows':7,'new_positive_repeat_rows':0,'heldout_inputs_read':0,
        'positive_changed':sum(r['positive_changed'] for r in audit),'prefix_changed':sum(r['prefix_changed'] for r in audit),
        'stop':sum(r['new_action']=='stop' for r in audit),'completion':sum(r['new_action']!='stop' for r in audit),'per_row':audit}
    (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)


if __name__=='__main__':main()

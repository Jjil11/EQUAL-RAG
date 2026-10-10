"""Frozen train-only KL scope and earliest-invalid-action labels. No test access."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
from pathlib import Path
from completion_pairs_20261010 import ROOT, encode_end, compiled_pair, repair, norm

RUN = ROOT / 'runs/qwen_priority_20261010'


def earliest_invalid(rejected, candidates, shared, stops):
    """A trie-compatible decision on the rejected path, before its first violation."""
    compatible = [c for c in candidates if c[:shared] == rejected[:shared]]
    if not compatible:
        raise ValueError('No positive shares the frozen context')
    for j in range(shared, len(rejected)):
        allowed = {c[j] for c in compatible if len(c) > j}
        if allowed.intersection(stops):
            allowed.update(stops)
        bad = rejected[j]
        if bad not in allowed:
            if not allowed:
                raise ValueError('Empty legal action set')
            return {'context_ids': rejected[:j], 'allowed_ids': sorted(allowed),
                    'rejected_id': bad, 'decision_position': j,
                    'steps_after_shared': j-shared}
        compatible = [c for c in compatible if len(c)>j and c[j] == bad]
        # Equivalent EOS is terminal, so a rejected completion must not reach it.
        if bad in stops:
            raise ValueError('Rejected path is a legal terminal alternative')
    raise ValueError('No illegal decision found')


def candidate_texts(row):
    spec=repair(row);prefix=spec['prefix'];missing=spec['missing_edges']
    if not missing:
        return [prefix, prefix+'.']
    values=set()
    for first in missing:
        edge=row['edges'][first]
        # Bounded aliases; punctuation/citation-bearing aliases are excluded.
        aliases=[edge['answer']]+sorted(set(edge['aliases']))
        aliases=list(dict.fromkeys(a.strip() for a in aliases if norm(a)
                    and '\n' not in a and '[' not in a and ']' not in a))[:8]
        tail=[f"{row['edges'][i]['answer']} [{row['edges'][i]['doc_id']}]"
              for i in missing if i != first]
        for alias in aliases:
            docs=sorted(set(edge.get('supporting_doc_ids',[edge['doc_id']])))
            for doc in docs:
                text=prefix+', '+', '.join([f'{alias} [{doc}]']+tail)
                values.update([text,text+'.'])
    return sorted(values)


def main():
    from transformers import AutoTokenizer, GenerationConfig
    import train_trajectory_equal as legacy
    source=ROOT/'runs/completion_repair_20261010/compiled_v3/qwen3_train.jsonl'
    expected='49bd1668c2d30eec0db716762e13079eb3eaa0f4bfbd15fbce12e6c7a53ffa21'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
    tok._equal_rag_disable_thinking=True
    stops=GenerationConfig.from_pretrained(tok.name_or_path).eos_token_id
    stops=[stops] if isinstance(stops,int) else stops
    out=RUN/'compiled';out.mkdir(parents=True,exist_ok=True)
    if (out/'train.jsonl').exists():raise FileExistsError('Frozen labels exist')
    audit=[]
    for row in rows:
        ids,_,labels=legacy.encode_example(tok,row['prompt'],row['short'],2048)
        ended,_,_=encode_end(tok,row['prompt'],row['short'],stops)
        if len(ended)>2048:raise ValueError('Original trajectory truncated')
        assert ids[:len(ended)].tolist()==ended.tolist()
        positions=(labels[1:]!=-100).nonzero().flatten().tolist()
        full=[p+1<len(ended) for p in positions]
        assert len(full)==len(row['retention_mask']) and any(full)
        row['full_retention_mask']=full
        row['failure_local_retention_mask']=row['retention_mask'] if row.get('has_trajectory') else full
        rec={'id':row['id'],'active':bool(row.get('has_trajectory')),
             'uniform_tokens':sum(row['retention_mask']),'full_tokens':sum(full),
             'failure_local_tokens':sum(row['failure_local_retention_mask'])}
        if row.get('has_trajectory'):
            _,rejected,n,spec=compiled_pair(tok,row,stops)
            assert spec==row['completion_pair']
            texts=candidate_texts(row);candidates=[]
            for text in texts:
                candidate,_,_=encode_end(tok,row['prompt'],text,stops)
                if len(candidate)>2048:raise ValueError('Candidate truncated')
                candidates.append(candidate.tolist())
            decision=earliest_invalid(rejected.tolist(),candidates,n,set(stops))
            assert decision['rejected_id'] not in decision['allowed_ids']
            assert decision['context_ids']==rejected[:decision['decision_position']].tolist()
            row['action_decision']=decision
            rec.update(action=spec['action'],candidate_count=len(candidates),
                       allowed_count=len(decision['allowed_ids']),
                       steps_after_shared=decision['steps_after_shared'])
        audit.append(rec)
    assert len(rows)==350 and sum(r['active'] for r in audit)==39
    target=out/'train.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    active=[r for r in rows if r.get('has_trajectory')]
    selected=[r for r in active if r['completion_pair']['action']=='stop']+[
        r for r in active if r['completion_pair']['action']!='stop'][:7]
    assert len(selected)==8
    (out/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    report={'rows':350,'active':39,'stop':1,'completion':38,'heldout_inputs_read':0,
            'source_sha256':expected,'compiled_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'eos_ids':stops,'counts':{k:sum(r[k] for r in audit) for k in
                ['uniform_tokens','full_tokens','failure_local_tokens']},'per_row':audit}
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)


if __name__=='__main__':main()

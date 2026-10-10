"""Audit all 350 training trajectories before any optimizer runs."""
import hashlib
import json
from pathlib import Path
from build_qwen_retrieval_align_20261010 import ROOT,RUN,CACHE
from completion_pairs_20261010 import covered,true_repeat


def main():
    rows=[];shards={}
    for i in range(4):
        assert (RUN/'mining'/f'shard_{i}_status.txt').read_text().strip()=='COMPLETE'
        p=RUN/'mining'/f'shard_{i}.jsonl';meta=json.loads((RUN/'mining'/f'shard_{i}_manifest.json').read_text())
        assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['shard_sha256']
        part=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        assert len(part)==meta['rows'] and all(r['global_index']%4==i for r in part)
        rows.extend(part);shards[str(i)]=meta
    rows.sort(key=lambda r:r['global_index'])
    original=[json.loads(x) for x in (RUN/'data/train_top5.jsonl').read_text().splitlines() if x.strip()]
    assert len(rows)==len({r['id'] for r in rows})==350 and [r['id'] for r in rows]==[r['id'] for r in original]
    records=[]
    for row,source in zip(rows,original):
        assert row['prompt']==source['prompt'] and row['edges']==source['edges'] and row['docs']==source['docs']
        assert row['anchor_output_tokens']==len(row['generation_token_ids'])
        mask=row['failure_local_retention_mask'];meta=row['cache_metadata'];p=Path(meta['path'])
        assert p.parent==CACHE and p.exists() and p.stat().st_size>0
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==meta['prompt_sha256']
        assert hashlib.sha256(row['short'].encode()).hexdigest()==meta['response_sha256']
        assert len(mask)==meta['shape'][0] and len(row['full_retention_mask'])==len(mask)
        if row['has_trajectory']:
            assert row['edges'] and row['verified_surface_repeat']
            assert len(covered(row['completion_pair']['chosen'],row['edges']))==len(row['edges'])
            assert not true_repeat({'short':row['completion_pair']['chosen']})
            assert row['action_decision']['rejected_id'] not in row['action_decision']['allowed_ids']
        else:assert mask==row['full_retention_mask']
        records.append({'id':row['id'],'active':row['has_trajectory'],'has_edges':bool(row['edges']),
                        'repeat':row['verified_surface_repeat'],'cap_hit':row['anchor_cap_hit'],
                        'retained_tokens':sum(mask),'original_tokens':len(mask),
                        'action':row.get('completion_pair',{}).get('action')})
    active=[r for r in rows if r['has_trajectory']]
    if len(active)<8:raise ValueError('Too few supported training faults for declared smoke; do not silently select other labels')
    out=RUN/'compiled'
    if out.exists():raise FileExistsError('Refusing frozen compilation overwrite')
    out.mkdir();target=out/'train.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    smoke=[r for r in active if r['completion_pair']['action']=='stop'][:4]
    smoke += [r for r in active if r['completion_pair']['action']!='stop'][:8-len(smoke)]
    if len(smoke)<8:smoke=active[:8]
    (out/'smoke.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in smoke))
    audit={'rows':350,'active':len(active),'stop':sum(r['action']=='stop' for r in records),
           'completion':sum(r['action']=='complete_then_stop' for r in records),
           'all_surface_repeat':sum(r['repeat'] for r in records),
           'uncertain_zero_edge_repeats_not_repaired':sum(r['repeat'] and not r['has_edges'] for r in records),
           'zero_lexical_support':sum(not r['has_edges'] for r in records),
           'anchor_cap_hits':sum(r['cap_hit'] for r in records),
           'retained_kl_tokens':sum(r['retained_tokens'] for r in records),
           'original_response_positions':sum(r['original_tokens'] for r in records),
           'cache_bytes':sum(Path(r['cache_metadata']['path']).stat().st_size for r in rows),
           'compiled_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
           'smoke_rows':len(smoke),'heldout_inputs_read':0,'shards':shards,'per_row':records}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k not in ['per_row','shards']},indent=2),flush=True)


if __name__=='__main__':main()

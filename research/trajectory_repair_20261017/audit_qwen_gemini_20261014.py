"""Verify Gemini's boundary proposal on the actual historical training states.

No historical labels overwritten. Natural termination and observed inventory
checks restrict diagnostic stop supervision, never certify semantic closure.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
from collections import Counter
import json
import qwen_equal_graph_20261014 as q
from compile_qwen_coherent_span_20261012 import common_prefix

SOURCE=q.ROOT/'runs/qwen_native_consistent_20261010/compiled/train.jsonl'
RUN=q.ROOT/'runs/qwen_gemini_audit_20261014'


def preflight():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True);records=[];native=[]
    for row in q.read(SOURCE):
        d=row.get('terminal_decision',{})
        if not row.get('has_trajectory') or d.get('origin')!='recorded_onpolicy_prefix':continue
        native.append(row);ctx=d['context_ids'];k=d['generation_prefix_tokens']
        prefix=tok.decode(row['generation_token_ids'][:k],skip_special_tokens=True).strip()
        full=tok.decode(ctx,skip_special_tokens=False)
        # Alter only the trailing delimiter, never remove a citation digit.
        closed=full.rstrip().rstrip(',').rstrip()
        candidates=[tok(closed+s,add_special_tokens=False)['input_ids'] for s in ['', '.', '\n']]
        roundtrip=tok(full,add_special_tokens=False)['input_ids']==ctx
        shared=min(common_prefix(ctx,x) for x in candidates)
        paths=[x+[151645] for x in candidates]
        assert all(p[:shared]==ctx[:shared] and len(p)>shared for p in paths)
        parsed_prefix=q.cited_units(prefix.rstrip().rstrip(',').rstrip())
        parsed_full=q.cited_units(row['short'])
        natural=bool(row.get('anchor_ended_by_eos',False))
        preserved=bool(parsed_prefix and parsed_full and
            {u['key'] for u in parsed_prefix}=={u['key'] for u in parsed_full})
        records.append({'id':row['id'],'original_context_ids':ctx,'rewound_context_ids':ctx[:shared],
            'positive_paths':paths,'original_prefix_tokens':k,'rewind_tokens':len(ctx)-shared,
            'last_token_id':ctx[-1],'last_token_text':tok.decode(ctx[-1:]),
            'first_allowed_ids':sorted({p[shared] for p in paths}),
            'first_rejected_id':ctx[shared] if shared<len(ctx) else row['generation_token_ids'][k],
            'context_roundtrip_exact':roundtrip,'teacher_naturally_ended':natural,
            'all_observed_units_already_in_prefix':preserved,
            'stop_diagnostic_eligible':natural and preserved and roundtrip,
            'no_semantic_closure_claim':True})
    assert len(native)==72
    assert not RUN.exists();RUN.mkdir();q.write_rows(RUN/'native_boundaries.jsonl',records)
    report={'native_sites':len(records),'source_sha256':q.sha(SOURCE),
        'rewind_histogram':dict(Counter(str(r['rewind_tokens']) for r in records)),
        'last_token_histogram':dict(Counter(r['last_token_text'] for r in records)),
        'roundtrip_mismatches':sum(not r['context_roundtrip_exact'] for r in records),
        'natural_end_trajectories':sum(r['teacher_naturally_ended'] for r in records),
        'observed_inventory_complete_at_prefix':sum(r['all_observed_units_already_in_prefix'] for r in records),
        'eligible_behavioral_stop_diagnostics':sum(r['stop_diagnostic_eligible'] for r in records),
        'variant_token_ids':{s:tok(s,add_special_tokens=False)['input_ids'] for s in [']','].',']\n','],']},
        'local_docs_snapshot':'Project GitHub release METHOD/ROADMAP describe Oct11 sparse version; Oct12 coherent-span experiments already exist.',
        'no_dev_or_sealed_reads':True,'code_sha256':q.sha(__file__)}
    q.dump(RUN/'preflight.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def probe():
    import torch
    import torch.nn.functional as F
    model,tok,mapping=q.load_model('anchor');model.eval();results=[]
    def stats(ids,allowed):
        x=torch.tensor(ids,device=model.get_input_embeddings().weight.device)[None]
        lp=F.log_softmax(model(input_ids=x,use_cache=False).logits[0,-1].float(),-1)
        pos=torch.logsumexp(lp[allowed],0)
        masked=lp.clone();masked[allowed]=-float('inf');bad=int(masked.argmax())
        best=int(lp.argmax())
        return {'argmax':best,'argmax_text':tok.decode([best]),'allowed_argmax':best in allowed,
            'allowed_mass':float(pos.exp()),'best_rejected':bad,'margin_to_best_nonallowed':float(pos-lp[bad])}
    with torch.inference_mode():
        for i,row in enumerate(q.read(RUN/'native_boundaries.jsonl')):
            original=stats(row['original_context_ids'],[151643,151645])
            boundary=stats(row['rewound_context_ids'],row['first_allowed_ids'])
            ends=[stats(p[:-1],[151643,151645]) for p in row['positive_paths']]
            results.append({'id':row['id'],'eligible':row['stop_diagnostic_eligible'],
                'original_eos':original,'boundary':boundary,'teacher_forced_ends':ends})
            q.dump(RUN/'probe_status.json',{'state':'PROBING','completed':i+1,'of':72})
    q.write_rows(RUN/'probe.jsonl',results)
    result={'state':'COMPLETE','n':72,'original_eos_argmax':sum(r['original_eos']['allowed_argmax'] for r in results),
        'legal_boundary_argmax':sum(r['boundary']['allowed_argmax'] for r in results),
        'any_teacher_forced_closure_eos_argmax':sum(any(x['allowed_argmax'] for x in r['teacher_forced_ends']) for r in results),
        'eligible_behavioral_stop_sites':sum(r['eligible'] for r in results),
        'warning':'Forced endpoint EOS is not free-generation reachability, semantic closure, or a success claim.',
        'adapter_sha256':q.sha(q.ANCHOR/'adapter_model.safetensors'),'placement':mapping,
        'code_sha256':q.sha(__file__)}
    q.dump(RUN/'probe_summary.json',result);q.dump(RUN/'probe_status.json',{'state':'COMPLETE'});print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    if a.probe:probe()
    else:preflight()

"""Exactly reuse the frozen scorer, ledger and 20,000-draw Q-only bootstrap."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import math
import numpy as np
from supp_runtime import *
import analyze_heldout311_20261015 as frozen
from score_strict_units_20261010 import score

VIEWS=['multi45','raw24','full311','multi','sparse']
EXPECTED={'multi45':(45,176),'raw24':(24,8),'full311':(311,484),'multi':(85,428),'sparse':(226,56)}

def source_predictions():
    dev={'base':SOURCE/'cached_eval/base.jsonl','base_block':SOURCE/'eval/base_block/predictions.jsonl',
         'anchor':SOURCE/'cached_eval/anchor.jsonl',
         'ce':PROJECT/'runs/qwen_tail_open_suite_20261015/stage1/eval/base_ce/predictions.jsonl',
         'a025':AONLY/'beta025/profile/eval69/predictions.jsonl',
         'a050':AONLY/'beta050/profile/eval69/predictions.jsonl'}
    return {name:read(path)+read(HELD/'predictions'/({'ce':'ce_s20261015','a025':'a025_s20261015','a050':'a050_s20261015'}.get(name,name))/'predictions.jsonl') for name,path in dev.items()}

def keys_for(inputs,view):
    return sorted(r['id'] for r in inputs if
        (view=='full311' and r['eval_split'] in ['multi','sparse']) or r['eval_split']==view)

def statistics(inputs,predictions,pairs,output,base_key):
    output.mkdir(parents=True,exist_ok=False)
    byid={r['id']:r for r in inputs}
    ps={m:{p['id']:p for p in rr} for m,rr in predictions.items()}
    assert all(len(predictions[m])==len(ps[m])==380 and set(ps[m])==set(byid) for m in ps)
    arrays={m:{} for m in ps};tables={m:{} for m in ps};audits={m:{} for m in ps};all_scores={}
    for m in ps:
        vals={}
        for key,p in ps[m].items():
            assert len(p['generated_token_ids'])==p['tokens']<=256
            assert p['hit_cap_without_eos']==(p['tokens']==256 and not p['ended_by_eos'])
            vals[key]=dict(score(byid[key],dict(p,grounded_edges=0)),true_cap=p['hit_cap_without_eos'])
        all_scores[m]=vals
        rows(output/'scores'/(m+'.jsonl'),[vals[k] for k in sorted(vals)])
        for view in VIEWS:
            keys=keys_for(inputs,view)
            a=frozen.sufficient([vals[k] for k in keys])
            assert (len(keys),int(a[:,1].sum()))==EXPECTED[view]
            arrays[m][view]=a
            tables[m][view]={'n':len(keys),'gold_edges':int(a[:,1].sum()),'grounded_edges':int(a[:,0].sum()),
                **dict(zip(frozen.FIELDS,frozen.observed(a)))}
            ll=[frozen.ledger(byid[k],ps[m][k],ps[base_key][k]) for k in keys]
            rows(output/'ledgers'/(m+'__'+view+'.jsonl'),ll)
            audits[m][view]=frozen.audit_sum(ll)
            assert audits[m][view]['locally_cited']['current']==int(a[:,0].sum())
            assert audits[m][view]['locally_cited']['net']==int(a[:,0].sum()-sum(all_scores[base_key][k]['grounded_edges'] for k in keys)) if base_key in all_scores else True
    cis={}
    for view in VIEWS:
        n=EXPECTED[view][0]
        ix=np.random.default_rng(20261010).integers(0,n,(20000,n))
        draws={m:frozen.bootstrap(a[view],ix) for m,a in arrays.items()}
        ob={m:frozen.observed(a[view]) for m,a in arrays.items()}
        for m,b in pairs:
            cis[m+'__minus__'+b+'__'+view]=frozen.interval(ob[m]-ob[b],draws[m]-draws[b],
                'supplementary_single_seed_paired_questions_Q_only')
    summary=frozen.clean({'tables':tables,'ledgers':audits,'paired_bootstrap':cis,
        'bootstrap_samples':20000,'bootstrap_seed':20261010,'single_training_seed':SEED,
        'all_comparisons_reported':True,'no_multiplicity_correction':True,
        'post_unsealing_supplement_not_new_blind_test':True,
        'prediction_sha256':{m:hashlib.sha256(json.dumps(predictions[m],sort_keys=True).encode()).hexdigest() for m in ps},
        'frozen_scorer_and_ledger_reused_without_modification':True})
    dump(output/'summary.json',summary)
    return summary

def package2():
    ps=source_predictions()
    archived={m:{r['id']:r for r in rr} for m,rr in ps.items()}
    registry=js(SUITE/'QWEN_OUTPUT_REGISTRY.json')
    replay={}
    for m,path in registry.items():
        rr=read(path)
        if m in archived:
            equal=sum(r['generated_token_ids']==archived[m][r['id']]['generated_token_ids'] for r in rr)
            replay[m]={'questions':len(rr),'identical_token_trajectories':equal,'different_token_trajectories':len(rr)-equal}
        ps[m]=rr
    dump(SUITE/'analysis/execution_replay_consistency.json',replay)
    new=['exhaustive_prompt','matched_sft','matched_dpo']
    pairs=[(m,'base') for m in ps if m!='base']
    pairs += [(m,b) for m in ['ce','a025'] for b in new]
    return statistics(eval_rows(),ps,pairs,SUITE/'analysis/package2','base')

def package3():
    paths={'base2':SUITE/'package3/base/eval/predictions380.jsonl'}
    paths.update({m:SUITE/'package3'/m/'eval/predictions380.jsonl' for m in ['matched_sft','equal_ce','equal_a025']})
    ps={m:read(p) for m,p in paths.items()}
    pairs=[(m,'base2') for m in ps if m!='base2']+[(m,'matched_sft') for m in ['equal_ce','equal_a025']]
    return statistics(eval_rows(),ps,pairs,SUITE/'analysis/package3','base2')

def fmt(x,scale=100):
    return 'N/A' if x is None or not math.isfinite(x) else f'{x*scale:.2f}'

def report(s2,s3):
    labels={'base':'Base Qwen3-14B','base_block':'Entity Blocker','anchor':'Historical Anchor',
        'ce':'EQUAL β=0','a025':'EQUAL β=0.25','a050':'EQUAL β=0.50',
        'exhaustive_prompt':'Exhaustive-Prompt Base','matched_sft':'Matched Whole-Answer SFT',
        'matched_dpo':'Matched Sequence DPO + 0.1 SFT','base2':'Base Mistral-7B',
        'equal_ce':'EQUAL β=0','equal_a025':'EQUAL β=0.25'}
    out=['# EQUAL-RAG Supplementary Results — 2026-10-17','',
      'Fixed seed 20261015. All newly trained adapters are unmerged, use 60 updates and 240 supervised record visits, and are evaluated regardless of outcome. The 311 questions were already unsealed in the main study; these are supplementary post-unsealing comparisons, not a second independent blind test.',
      '', '## Protocol and interpretive limits','',
      '- Same frozen lexical scorer, normalization, gain/loss ledger, and Q-only paired question bootstrap; 20,000 resamples, seed 20261010, nominal unadjusted 95% intervals.',
      '- Whole-answer CE covers every response token. DPO uses summed sequence log probabilities, β_DPO=0.1, frozen Base reference, and 0.1 whole-answer CE stabilization. The 20 C2 pairs are identical; they receive stabilization CE but no artificial preference labels.',
      '- Optimizer updates, supervised positive targets, LoRA initialization and 240-record visit order are matched. EQUAL additionally uses 240 auxiliary retention visits; equal updates are not a claim of equal FLOPs or wall-clock cost.',
      '- Retention for adapted models uses the full original teacher-forced contexts. For Exhaustive Base, an additional changed-prompt counterfactual is reported separately; it is not a same-prompt parameter-retention statistic.',
      '- Tokenization mismatch does not prove universal inability to continue, a semantic evidence-exhaustion detector, or guaranteed benefit on every backbone.',
      '', '## Table S1 — tokenizer boundary audit','',
      '| Tokenizer | Literal `]` IDs | Literal `],` IDs | Contextual native `]` ID | Contextual continuation first changed ID | Mismatch |',
      '|---|---|---|---:|---:|---:|']
    audit=js(SUITE/'analysis/tokenizer_boundary_audit.json')
    seen=set()
    for r in audit['results']:
        if r['tokenizer'].startswith('local:'):continue
        key=r['tokenizer']
        if key in seen:continue
        seen.add(key)
        a=r['context_native_example_ids'];b=r['context_continuation_example_ids']
        k=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
        out.append(f"| {key} | {r['literal_ids'][']']} | {r['literal_ids']['],']} | {a[k] if k<len(a) else 'END'} | {b[k] if k<len(b) else 'END'} | {r['mismatches']}/21 ({r['boundary_mismatch_rate_percent']:.2f}%) |")
    out+=['','Context IDs refer to `Alice [1], Bob [1]` versus its continuation. Standalone IDs can differ from in-context IDs (notably Mistral). Qwen2.5 and Qwen3 are not independent tokenizer families. A second local Mistral copy is a consistency check, not another family. Unavailable/gated Llama3 and Gemma are not counted as zero mismatches. Exact windows and provenance are archived.','']
    for number,s in [(2,s2),(3,s3)]:
        out += [f'## Table S{number} — '+('Qwen3-14B matched-budget comparisons' if number==2 else 'Mistral-7B cross-family transfer'),'']
        for view in VIEWS:
            out += [f'### {view}: {EXPECTED[view][0]} questions / {EXPECTED[view][1]} support groups','',
                '| System | Q % | Full F1 % | Tokens | Repetition % | True cap % | G | L | Net | New name + cite | Citation fix |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for m,t in s['tables'].items():
                v=t[view];a=s['ledgers'][m][view];c=a['locally_cited'];d=a['Q_decomposition']
                values=[fmt(v[f],1 if f=='mean_tokens' else 100) for f in frozen.FIELDS]
                out.append('| '+labels[m]+' | '+' | '.join(values)+f" | {c['added']} | {c['lost']} | {c['net']:+d} | {d['new_name_with_supported_citation']} | {d['old_name_citation_repaired']} |")
            out += ['', '| Paired comparison | ΔQ pp [95% CI] | ΔFull F1 pp [95% CI] | Defined Q draws / 20,000 |',
                    '|---|---|---|---:|']
            for key,c in s['paired_bootstrap'].items():
                m,b,v=key.split('__minus__')[0],None,None
                rest=key.split('__minus__')[1];b,v=rest.rsplit('__',1)
                if v!=view:continue
                cells=[]
                for f in ['grounded_quality','answer_f1']:
                    x=c[f];cells.append(f"{fmt(x['delta'])} [{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}]")
                out.append(f"| {labels[m]} − {labels[b]} | "+' | '.join(cells)+f" | {c['grounded_quality']['defined_draws']} |")
            out.append('')
    out += ['## Training and retention diagnostics','', '### Qwen3-14B','', '| System | Updates | Supervised visits | Auxiliary retention visits | Full-context Ret./120 | Training seconds |','|---|---:|---:|---:|---:|---:|']
    for m in ['matched_sft','matched_dpo']:
        meta=js(SUITE/'single_gpu_v3/package2'/m/'manifest.json')
        ret=js(SUITE/'single_gpu_v3/package2'/m/'retention_summary.json')
        out.append(f"| {labels[m]} | 60 | 240 | 0 | {ret['exact']}/120 | {meta['training_seconds']:.1f} |")
    out += ['', '### Mistral-7B','', 'Native compilation counts: `'+json.dumps(js(SUITE/'package3/compiled/counts.json'),ensure_ascii=False)+'`.','',
        '| System | A first action | First new entity exact tokens | C1 closure | C2 stop | Retained trajectories |','|---|---:|---:|---:|---:|---:|']
    for m in ['base','matched_sft','equal_ce','equal_a025']:
        d=js(SUITE/'package3'/m/'diagnostic/training_diagnostics_summary.json')
        ret=js(SUITE/'package3'/m/'diagnostic/retention_summary.json')
        out.append(f"| {m} | {d['A_first_action']}/{d['A_total']} | {d['A_first_entity_exact']}/{d['A_total']} | {d['C1']}/{d['A_total']} | {d['C2']}/{d['C2_total']} | {ret['exact']}/{ret['trajectories']} |")
    out += ['', 'Second-backbone identity: `'+MISTRAL+'`. All weight/config/tokenizer file SHA-256 digests are in `package3/model_identity.json`. The 350 training rollouts, eligible/rejected label audit, deterministic caps, compiled records, full-prefix references and visit schedules are archived.',
        '', '## Verification & checksums','', 'All frozen source snapshots are checked before and after; no original checkpoint, prediction file, or main blind-test report is overwritten. Raw prediction digests:','',
        '| New prediction file | Rows | SHA-256 |','|---|---:|---|']
    checks={}
    for p in sorted(SUITE.rglob('*predictions*.jsonl')):
        if '/analysis/' in str(p):continue
        rel=str(p.relative_to(SUITE));check={'sha256':sha(p),'rows':len(read(p))};checks[rel]=check
        out.append(f"| `{rel}` | {check['rows']} | `{check['sha256']}` |")
    protected=js(SUITE/'protected_before.json')
    for path,rec in protected.items():assert sha(path)==rec['sha256'],path
    dump(SUITE/'analysis/verification.json',{'status':'PASS','unchanged_frozen_files':len(protected),
         'predictions':checks,'all_requested_view_sizes_and_denominators_verified':True,
         'all_ledger_gain_loss_identities_verified':True,'main_report_never_written':True,
         'frozen_scorer_ledger_bootstrap_reused':True})
    out += ['', f"Verification PASS: {len(protected)} protected frozen files unchanged; all requested sample counts/denominators and gain/loss identities checked. See `analysis/verification.json` and raw ledgers for full detail.",
        '', 'All positive and negative outcomes are retained. This single-seed supplementary study does not add three-seed evidence for the new baselines or Mistral; no universal cross-model claim follows from one additional backbone.',
        '', 'DPO source: https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html',
        'Mistral model card: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3','']
    replay=js(SUITE/'analysis/execution_replay_consistency.json')
    out += ['', '## Execution-placement reproducibility audit','',
        'Cross-device attempts stalled or produced non-finite/OOM failures. All original attempts and logs are preserved. Final Qwen evaluation uses one full A100 per process, BF16/SDPA and unmerged adapters. Hyperparameters and final checkpoints are unchanged. The completed 60-step DPO adapter is reused; the non-finite SFT attempt completed zero optimizer updates and is not a competing checkpoint.', '',
        '| Frozen system replay | Identical token paths / 380 | Different token paths |',
        '|---|---:|---:|']
    for m,v in replay.items():
        out.append(f"| {labels[m]} | {v['identical_token_trajectories']}/380 | {v['different_token_trajectories']} |")
    out += ['', 'Table S2 uses the replayed Base and frozen EQUAL adapters so the new-baseline causal comparisons use the same execution placement. Historical Anchor and Entity Blocker reuse original archived predictions; these are not matched-budget causal controls. Original frozen predictions and the original 311-question report remain untouched. Non-identical replays are disclosed, not used to select checkpoints or hyperparameters.','']
    er=js(SUITE/'single_gpu_v3/package2/exhaustive_prompt/cross_prompt_retention/retention_summary.json')
    out.append(f"Exhaustive-Prompt cross-prompt counterfactual retention: {er['exact']}/{er['trajectories']} full original continuations retain every reference top-1 action when teacher-forced under the modified prompt. This is not identical-prompt parameter retention.")
    out += ['', '| Frozen EQUAL replay | Full-context Ret./120 |','|---|---:|']
    for m in ['base','ce','a025','a050']:
        er=js(SUITE/'single_gpu_v3/replay'/m/'diagnostic/retention_summary.json')
        out.append(f"| {labels[m]} | {er['exact']}/{er['trajectories']} |")
    destination=PROJECT/'EQUAL_SUPPLEMENTARY_RESULTS_20261017.md'
    assert not destination.exists()
    destination.write_text('\n'.join(out),encoding='utf-8')
    copy=SUITE/'analysis/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md'
    with safe(copy).open('x',encoding='utf-8') as f:f.write('\n'.join(out))
    print(json.dumps({'report':str(destination),'sha256':sha(destination)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['package2','package3','report']);a=ap.parse_args()
    if a.task=='package2':package2()
    elif a.task=='package3':package3()
    else:report(js(SUITE/'analysis/package2/summary.json'),js(SUITE/'analysis/package3/summary.json'))

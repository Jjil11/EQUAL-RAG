"""Authoritative full-answer F1, true-EOT and tail audit after fixed evaluations."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from score_strict_units_20261010 import score,summarize,paired

ROOT=Path(str(_release_root))
RUN=ROOT/'runs/submission_20261010'
ARMS={'qwen3':['baseline','semantic_eos','unit_ul','wrong_boundary','kl_only'],
      'mistral':['baseline','semantic_eos','unit_ul']}


def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def relative_utility(method,base):
    m={x['id']:x for x in method};b={x['id']:x for x in base};keys=sorted(m)
    def arr(items):return np.array([[items[k]['grounded_edges'],items[k]['gold_edges'],items[k]['tokens']] for k in keys],float)
    a,z=arr(m),arr(b)
    def util(v):return (v[...,0].sum(-1)/np.maximum(1,v[...,1].sum(-1)))/np.maximum(1,v[...,2].mean(-1))
    observed=util(a)/max(1e-12,util(z))-1
    ix=np.random.default_rng(20261010).integers(0,len(keys),(20000,len(keys)))
    boots=util(a[ix])/np.maximum(1e-12,util(z[ix]))-1
    return {'relative_delta':float(observed),'ci95':np.quantile(boots,[.025,.975]).tolist()}


def main():
    while True:
        p=RUN/'finalization_status.json'
        state=json.loads(p.read_text()).get('status') if p.exists() else None
        if state=='FAILED':raise RuntimeError('Upstream evaluation failed')
        if state=='COMPLETE':break
        time.sleep(15)
    source=ROOT/'ALCE/data/qampari_eval_gtr_top100.json'
    official={str(x.get('id',x.get('sample_id',i))):x['answers'] for i,x in enumerate(json.loads(source.read_text()))}
    report={'status':'COMPLETE_AUTOMATED_DEVELOPMENT_ROUND','n_models':2,'training_seed':20260930,
            'sealed_holdout_predictions':0,'human_annotations_completed':0,
            'official_answer_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'metric_warning':'Full-gold ALCE-style answer F1; exact-unit lexical citation proxy, not NLI or human entailment.',
            'control_warning':'wrong_boundary uses retokenized earlier text boundaries, unlike exact-token-aligned semantic_eos; not sufficient alone for a clean causal localization claim.',
            'families':{}}
    for family,arms in ARMS.items():
        folder=RUN/f'{family}_frozen_comparison'
        report['families'][family]={}
        for setting in ['regression_dev','prospective_dev']:
            source_data=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
            data={x['id']:x for x in read(source_data)}
            if setting=='regression_dev':
                for k,row in data.items():row['all_gold_answers']=official[k]
            results={arm:[score(data[x['id']],x) for x in read(folder/setting/arm/'eval_results.jsonl')] for arm in arms}
            telemetry={arm:{x['id']:x for x in read(folder/setting/arm/'termination_telemetry.jsonl')} for arm in arms}
            table={arm:summarize(rows) for arm,rows in results.items()}
            for arm,rows in results.items():
                table[arm]['true_cap_hit_rate']=float(np.mean([telemetry[arm][x['id']]['hit_cap_without_eos'] for x in rows]))
                table[arm]['token_quantiles']=dict(zip(['p50','p90','p95','p99'],np.quantile([x['tokens'] for x in rows],[.5,.9,.95,.99]).tolist()))
                zeros=[x for x in rows if not data[x['id']]['edges']]
                table[arm]['zero_lexical_support_n']=len(zeros)
                table[arm]['zero_support_mean_tokens']=float(np.mean([x['tokens'] for x in zeros])) if zeros else None
                table[arm]['zero_support_f1']=float(np.mean([x['answer_f1'] for x in zeros])) if zeros else None
                (folder/setting/f'{arm}_full_gold_scored.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
            comparisons={}
            for arm in arms:
                if arm=='baseline':continue
                item=paired(results[arm],results['baseline'])
                item['relative_utility']=relative_utility(results[arm],results['baseline'])
                for stratum,keep in [('baseline_true_cap',lambda x:telemetry['baseline'][x['id']]['hit_cap_without_eos']),
                                     ('baseline_noncap',lambda x:not telemetry['baseline'][x['id']]['hit_cap_without_eos'])]:
                    ids={x['id'] for x in results['baseline'] if keep(x)}
                    if ids:
                        item[stratum]={'n':len(ids),'paired':paired([x for x in results[arm] if x['id'] in ids],
                            [x for x in results['baseline'] if x['id'] in ids],samples=10000)}
                    else:item[stratum]={'n':0,'paired':None}
                comparisons[f'{arm}_minus_baseline']=item
            comparisons['semantic_eos_minus_unit_ul']=paired(results['semantic_eos'],results['unit_ul'])
            comparisons['semantic_eos_minus_unit_ul']['relative_utility']=relative_utility(results['semantic_eos'],results['unit_ul'])
            if 'kl_only' in arms:
                comparisons['semantic_eos_minus_kl_only']=paired(results['semantic_eos'],results['kl_only'])
            report['families'][family][setting]={'table':table,'comparisons':comparisons}
    (RUN/'development_round_summary_all_answers.json').write_text(json.dumps(report,indent=2))
    files=[p for p in RUN.rglob('*') if p.is_file() and 'adapter' not in p.parts and p.suffix in {'.json','.jsonl','.txt','.tsv'}
           and not p.name.startswith('artifact_hashes') and 'PRIVATE' not in p.name]
    (RUN/'artifact_hashes_all_answers.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    (RUN/'all_answers_finalization_status.json').write_text(json.dumps({'status':'COMPLETE','holdout_inference':False,'human_annotations':0},indent=2))
    print('Authoritative full-answer, paired, tail audit complete',flush=True)


if __name__=='__main__':main()

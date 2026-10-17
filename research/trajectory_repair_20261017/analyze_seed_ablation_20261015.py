"""Question-paired fixed-seed and two-level Bootstrap; never ensemble outputs."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import shutil
import numpy as np
import analyze_tail_open_suite_20261015 as old
import qwen_seed_ablation_20261015 as r

q=r.q;RUN=r.RUN;SOURCE=r.SOURCE
FIELDS=old.FIELDS


def sufficient(vals):
    return np.array([[v['grounded_edges'],v['gold_edges'],v['tokens'],v['answer_f1'],float(v['repeated_units']>0),float(v['cap_hit_proxy'])] for v in vals],float)


def observed(a):
    return np.array([a[:,0].sum()/a[:,1].sum(),a[:,3].mean(),a[:,2].mean(),a[:,4].mean(),a[:,5].mean()])


def sampled(a,ix):
    v=a[ix];den=v[:,:,1].sum(1)
    return np.column_stack([np.divide(v[:,:,0].sum(1),den,out=np.full(len(ix),np.nan),where=den>0),
        v[:,:,3].mean(1),v[:,:,2].mean(1),v[:,:,4].mean(1),v[:,:,5].mean(1)])


def hierarchical(a,qix,six):
    # Shared sampled seed indices for paired arms. Baseline repeated across seeds.
    v=a[six[:,:,None],qix[:,None,:],:].mean(1)
    den=v[:,:,1].sum(1)
    return np.column_stack([np.divide(v[:,:,0].sum(1),den,out=np.full(len(qix),np.nan),where=den>0),
        v[:,:,3].mean(1),v[:,:,2].mean(1),v[:,:,4].mean(1),v[:,:,5].mean(1)])


def interval(obs,draw,kind):
    out={f:{'delta':float(obs[i]),'ci95':np.nanquantile(draw[:,i],[.025,.975]).tolist()} for i,f in enumerate(FIELDS)}
    out.update(samples=len(draw),seed=20261010,seed_draw_seed=20261011,
        zero_Q_denominator_draws=int(np.isnan(draw[:,0]).sum()),kind=kind,
        no_multiple_comparison_adjustment=True,Q_undefined_draws_excluded_only_from_Q=True)
    return out


def sources():
    paths={'base':SOURCE/'cached_eval/base.jsonl','base_block':SOURCE/'eval/base_block/predictions.jsonl',
        'anchor':SOURCE/'cached_eval/anchor.jsonl','historical_pair':r.prev.OPEN/'stage1/eval/base_pair/predictions.jsonl'}
    full={}
    for config in r.CONFIGS:
        full[config]={}
        for seed in r.SEEDS:
            key=f'{config}_s{seed}';full[config][seed]=key
            if seed==r.SEEDS[0]:
                path=r.prev.OPEN/'stage1/eval/base_ce/predictions.jsonl' if config=='ce' else r.AONLY/('beta025' if config=='a025' else 'beta050')/'profile/eval69/predictions.jsonl'
            else:path=RUN/f'{config}_seed{seed}'/'profile/eval69/predictions.jsonl'
            paths[key]=path
    paths.update({task:RUN/task/'profile/eval69/predictions.jsonl' for task,s in r.TASKS.items() if s['variant']!='full'})
    return paths,full


def analyze(stage1=False):
    if stage1:
        for task,spec in r.TASKS.items():
            if spec['variant']=='full':assert json.loads((RUN/task/'status.json').read_text())['state']=='COMPLETE'
    else:assert json.loads((RUN/'status.json').read_text())['state']=='ALL_GENERATIONS_COMPLETE'
    folder=RUN/('analysis_multiseed_stage' if stage1 else 'analysis')
    pred_dir=RUN/('predictions_multiseed_stage' if stage1 else 'predictions_all')
    folder.mkdir(exist_ok=False);(folder/'ledgers').mkdir();pred_dir.mkdir()
    inputs=q.read(RUN/'data/eval69.jsonl');byid={v['id']:v for v in inputs}
    train=q.read(RUN/'data/training_source350_snapshot.jsonl')
    from entity_abc_contracts_20261015 import normalize
    assert not set(byid)&{v['id'] for v in train}
    assert not {normalize(v['question']) for v in inputs}&{normalize(v['question']) for v in train}
    paths,full=sources();preds={};scored={};tables={};audits={}
    if stage1:
        allowed={'base','base_block','anchor','historical_pair'}|{key for mapping in full.values() for key in mapping.values()}
        paths={key:path for key,path in paths.items() if key in allowed}
    historical={v['id']:v for v in q.read(old.c.PREVIOUS/'eval/base_pair/predictions.jsonl')}
    if not (RUN/'historical100_Pair_predictions69.jsonl').exists():q.write_rows(RUN/'historical100_Pair_predictions69.jsonl',list(historical.values()))
    for model,path in paths.items():
        ps=q.read(path);assert len(ps)==69 and len({v['id'] for v in ps})==69
        assert {v['id'] for v in ps}==set(byid)
        preds[model]={v['id']:v for v in ps}
        shutil.copy2(path,pred_dir/(model+'.jsonl'))
        for p in ps:
            assert p['prompt_token_ids']==preds['base'][p['id']]['prompt_token_ids']
            assert p['tokens']==len(p['generated_token_ids'])
            assert p['hit_cap_without_eos']==(p['tokens']==256 and p['generated_token_ids'][-1] not in r.c.STOPS)
            if model not in ['base','base_block','anchor']:assert not p['blocking_events']
        vals=[dict(old.score(byid[p['id']],dict(p,grounded_edges=0)),cap_hit_proxy=p['hit_cap_without_eos']) for p in ps]
        scored[model]={v['id']:v for v in vals};q.write_rows(folder/(model+'_scored69.jsonl'),vals)
        tables[model],audits[model]={},{}
        for split in ['multi45','raw24']:
            vv=[v for v in vals if byid[v['id']]['eval_split']==split]
            tables[model][split]=dict(old.summarize(vv),true_cap_hit_rate=sum(v['cap_hit_proxy'] for v in vv)/len(vv))
            audits[model][split]=old.audit(model,preds[model],preds['base'],historical,byid,split,folder/'ledgers')
            assert audits[model][split]['locally_cited']['current']==round(tables[model][split]['grounded_quality']*(176 if split=='multi45' else 8))
    comparisons=[(m,'base') for m in paths if m!='base']
    if not stage1:comparisons += [(task,full[s['config']][r.SEEDS[0]]) for task,s in r.TASKS.items() if s['variant']!='full']
    comparisons += [(full[config][seed],full['ce'][seed]) for seed in r.SEEDS for config in ['a025','a050']]
    comparisons += [(full['a050'][seed],full['a025'][seed]) for seed in r.SEEDS]
    comparisons += [('historical_pair',full['ce'][r.SEEDS[0]])]
    comparisons=list(dict.fromkeys(comparisons))
    cis={};mean_cis={};hier_cis={};means={};mean_audits={}
    mean_pairs=[(cfg,'base') for cfg in r.CONFIGS]+[('a025','ce'),('a050','ce'),('a050','a025')]
    for split in ['multi45','raw24']:
        ids=sorted(k for k,v in byid.items() if v['eval_split']==split)
        ix=np.random.default_rng(20261010).integers(0,len(ids),(20000,len(ids)))
        six=np.random.default_rng(20261011).integers(0,3,(20000,3))
        arrays={m:sufficient([scored[m][k] for k in ids]) for m in paths}
        ob={m:observed(a) for m,a in arrays.items()};dr={m:sampled(a,ix) for m,a in arrays.items()}
        for m,b in comparisons:
            cis[m+'__minus__'+b+'__'+split]=interval(ob[m]-ob[b],dr[m]-dr[b],'paired_questions_single_seed')
        stacks={cfg:np.stack([arrays[full[cfg][seed]] for seed in r.SEEDS]) for cfg in r.CONFIGS}
        stacks['base']=np.repeat(arrays['base'][None],3,axis=0)
        mob={cfg:observed(a.mean(0)) for cfg,a in stacks.items()}
        mdr={cfg:sampled(a.mean(0),ix) for cfg,a in stacks.items()}
        hdr={cfg:hierarchical(a,ix,six) for cfg,a in stacks.items()}
        for cfg in r.CONFIGS:
            means.setdefault(cfg,{})[split]={f:{'mean':float(np.mean([ob[full[cfg][seed]][i] for seed in r.SEEDS])),
                'SD_sample_ddof1':float(np.std([ob[full[cfg][seed]][i] for seed in r.SEEDS],ddof=1))} for i,f in enumerate(FIELDS)}
            for i,f in enumerate(FIELDS):assert abs(means[cfg][split][f]['mean']-mob[cfg][i])<1e-12
            mean_audits.setdefault(cfg,{})[split]={}
            for mode in ['name_only','locally_cited','Q_decomposition']:
                mean_audits[cfg][split][mode]={}
                for field in audits[full[cfg][r.SEEDS[0]]][split][mode]:
                    vv=[audits[full[cfg][seed]][split][mode][field] for seed in r.SEEDS]
                    mean_audits[cfg][split][mode][field]={'mean':float(np.mean(vv)),'SD_sample_ddof1':float(np.std(vv,ddof=1))}
            # Group-wise frequency, not union/voting/rewritten answers.
            freqs=[]
            for k in ids:
                input_row=byid[k];bn=old.membership(input_row,preds['base'][k],False);bc=old.membership(input_row,preds['base'][k],True)
                counts={name:{} for name in ['new_name_with_supported_citation','old_name_citation_repaired','lost_supported_group']}
                for seed in r.SEEDS:
                    nh=old.membership(input_row,preds[full[cfg][seed]][k],False);ch=old.membership(input_row,preds[full[cfg][seed]][k],True)
                    sets={'new_name_with_supported_citation':ch-bn,'old_name_citation_repaired':(ch&bn)-bc,'lost_supported_group':bc-ch}
                    for name,vv in sets.items():
                        for entity in vv:counts[name][str(entity)]=counts[name].get(str(entity),0)+1
                freqs.append({'id':k,'seed_count':3,'counts_0_to_3':counts,'edges':input_row['edges']})
            q.write_rows(folder/'ledgers'/(cfg+'__seed_frequency__'+split+'.jsonl'),freqs)
        for m,b in mean_pairs:
            key=m+'__minus__'+b+'__'+split
            mean_cis[key]=interval(mob[m]-mob[b],mdr[m]-mdr[b],'paired_questions_mean_of_fixed_3_seeds')
            hier_cis[key]=interval(mob[m]-mob[b],hdr[m]-hdr[b],'paired_seeds_and_questions_two_level_bootstrap_3_seeds_only')
    influence={}
    for m in paths:
        rr=q.read(folder/'ledgers'/(m+'__multi45.jsonl'))
        net=[len(v['locally_cited']['added'])-len(v['locally_cited']['lost']) for v in rr]
        influence[m]={'positive_questions':sum(v>0 for v in net),'negative_questions':sum(v<0 for v in net),
            'largest_gain':max(net),'net_without_largest_gain':sum(net)-max(net)}
    out={'models':list(paths),'full_config_seed_keys':{cfg:{str(s):v for s,v in mapping.items()} for cfg,mapping in full.items()},
        'tables':tables,'name_audits':audits,'three_seed_mean_SD':means,'three_seed_mean_audits':mean_audits,
        'paired_bootstrap':cis,'three_seed_mean_paired_bootstrap':mean_cis,'two_level_paired_bootstrap':hier_cis,
        'question_influence45':influence,'seed_list':r.SEEDS,'bootstrap_samples':20000,
        'single_variable_ablation_specs':{k:s for k,s in r.TASKS.items() if s['variant']!='full'} if not stage1 else {},
        'stage':'multiseed_only_while_frozen_ablations_continue' if stage1 else 'all_jobs_complete',
        'no_ensemble_text_or_voting':True,'all_new_models_unmerged':True,'no_dev_selection':True,
        'reused_development_not_blind':True,'sealed311_reads':0,'script_sha256':q.sha(__file__)}
    q.dump(folder/'summary.json',out)
    print(json.dumps({'models':len(paths),'new_jobs':len(r.TASKS),'three_seed_mean_SD':means,
        'single_seed_comparisons':len(cis),'fixed_three_seed_comparisons':len(mean_cis),'hierarchical_comparisons':len(hier_cis)},indent=2),flush=True)
    return out


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--multiseed-only',action='store_true');z=ap.parse_args()
    analyze(z.multiseed_only)

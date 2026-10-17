"""Frozen held-out scoring, matched Seed statistics, report and SHA archive."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tarfile
import time
import numpy as np
import qwen_heldout311_20261015 as r
from score_strict_units_20261010 import norm, score

FIELDS=['grounded_quality','answer_f1','mean_tokens','repetition_rate','true_cap_hit_rate']
LABEL={'base':'原始Qwen','base_block':'Qwen＋防重探针','anchor':'旧Anchor','ce':'CE β=0','a025':'AOnly β=0.25','a050':'AOnly β=0.50'}
VIEWS={'full311':'Full311 全集','multi':'多答案词面支持子集（≥2组）','sparse':'稀疏／单答案词面支持子集（0–1组）'}


def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [clean(v) for v in obj]
    if isinstance(obj,(float,np.floating)):return float(obj) if math.isfinite(obj) else None
    if isinstance(obj,np.integer):return int(obj)
    return obj


def save(p,obj):r.dump(p,clean(obj),True)


def members(row,pred,cited=False):
    units=[u.strip() for u in pred['output'].rstrip().rstrip('.').rstrip(',').split(',') if norm(u)]
    return {i for i,g in enumerate(row['edges']) if any(norm(u) in {norm(a) for a in g['aliases']} and
        (not cited or set(g.get('supporting_doc_ids',[g['doc_id']]))&{int(v) for v in re.findall(r'\[(\d+)\]',u)}) for u in units)}


def ledger(row,cur,base):
    n,c=members(row,cur),members(row,cur,True);bn,bc=members(row,base),members(row,base,True)
    out={'id':row['id'],'edges':row['edges']}
    for name,a,b in [('name_only',n,bn),('locally_cited',c,bc)]:
        out[name]={'base':sorted(b),'current':sorted(a),'added':sorted(a-b),'lost':sorted(b-a),
                   'added_names':[row['edges'][i]['aliases'] for i in sorted(a-b)],'lost_names':[row['edges'][i]['aliases'] for i in sorted(b-a)]}
    out['Q_decomposition']={'new_name_with_supported_citation':sorted(c-bn),'old_name_citation_repaired':sorted((c&bn)-bc),
        'original_supported_name_missing':sorted(bc-n),'name_retained_citation_lost':sorted((bc&n)-c)}
    assert len(c-bc)==len(c-bn)+len((c&bn)-bc)
    assert len(bc-c)==len(bc-n)+len((bc&n)-c)
    return out


def audit_sum(rr):
    out={}
    for name in ['name_only','locally_cited']:
        out[name]={field:sum(len(v[name][field]) for v in rr) for field in ['base','current','added','lost']}
        out[name]['net']=out[name]['added']-out[name]['lost']
    out['Q_decomposition']={field:sum(len(v['Q_decomposition'][field]) for v in rr) for field in
        ['new_name_with_supported_citation','old_name_citation_repaired','original_supported_name_missing','name_retained_citation_lost']}
    return out


def sufficient(vals):
    return np.array([[v['grounded_edges'],v['gold_edges'],v['tokens'],v['answer_f1'],float(v['repeated_units']>0),float(v['true_cap'])] for v in vals],float).reshape(-1,6)


def observed(a):
    if len(a)==0:return np.full(5,np.nan)
    den=a[:,1].sum()
    return np.array([a[:,0].sum()/den if den else np.nan,a[:,3].mean(),a[:,2].mean(),a[:,4].mean(),a[:,5].mean()])


def reduce_draw(v):
    den=v[:,:,1].sum(1)
    return np.column_stack([np.divide(v[:,:,0].sum(1),den,out=np.full(len(v),np.nan),where=den>0),v[:,:,3].mean(1),v[:,:,2].mean(1),v[:,:,4].mean(1),v[:,:,5].mean(1)])


def bootstrap(a,qix,six=None,chunk=256):
    if qix.shape[1]==0:return np.full((len(qix),5),np.nan)
    out=[]
    for start in range(0,len(qix),chunk):
        ix=qix[start:start+chunk]
        v=a[ix] if six is None else a[six[start:start+chunk,:,None],ix[:,None,:],:].mean(1)
        out.append(reduce_draw(v))
    return np.concatenate(out)


def interval(delta,draw,kind):
    out={}
    for i,f in enumerate(FIELDS):
        finite=draw[np.isfinite(draw[:,i]),i]
        out[f]={'delta':float(delta[i]),'ci95':np.quantile(finite,[.025,.975]).tolist() if len(finite) else [None,None],
                'defined_draws':len(finite)}
    out.update(samples=len(draw),question_rng_seed=20261010,seed_rng_seed=20261011,kind=kind,
        zero_Q_denominator_draws=int(np.isnan(draw[:,0]).sum()),no_multiple_comparison_adjustment=True,
        undefined_Q_excluded_only_from_Q=True)
    return clean(out)


def mean_record(v):
    return {'mean':float(np.mean(v)),'SD_sample_ddof1':float(np.std(v,ddof=1))}


def statistics(inputs,predictions,folder):
    """Pure scoring/statistics of frozen input/prediction mappings; no model access."""
    folder=Path(folder);folder.mkdir(exist_ok=False);(folder/'ledgers').mkdir()
    byid={v['id']:v for v in inputs};models=list(r.systems());assert list(predictions)==models
    vals={};tables={};audits={};arrays={};ledgers={}
    for model in models:
        ps=predictions[model];assert set(ps)==set(byid)
        vals[model]={};tables[model]={};audits[model]={};arrays[model]={};ledgers[model]={}
        for k,row in byid.items():
            p=ps[k];ids=p['generated_token_ids']
            assert p['prompt_token_ids']==row['prompt_token_ids'] and p['tokens']==len(ids)<=256 and len(ids)>0
            assert p['ended_by_eos']==(ids[-1] in r.STOPS)
            assert p['hit_cap_without_eos']==(len(ids)==256 and ids[-1] not in r.STOPS)
            if model!='base_block':assert not p['blocking_events']
            vals[model][k]=dict(score(row,dict(p,grounded_edges=0)),true_cap=p['hit_cap_without_eos'])
        r.rows(folder/(model+'_scored311.jsonl'),list(vals[model].values()))
        for view in r.VIEWS:
            keys=sorted(k for k,v in byid.items() if view=='full311' or v['eval_split']==view)
            a=sufficient([vals[model][k] for k in keys]);arrays[model][view]=a;ob=observed(a)
            tables[model][view]={'n':len(keys),'Q_effective_questions':sum(bool(byid[k]['edges']) for k in keys),
                'gold_edges':int(a[:,1].sum()),'grounded_edges':int(a[:,0].sum()),**dict(zip(FIELDS,ob))}
            rr=[ledger(byid[k],ps[k],predictions['base'][k]) for k in keys];ledgers[model][view]=rr
            r.rows(folder/'ledgers'/(model+'__'+view+'.jsonl'),rr);audits[model][view]=audit_sum(rr)
            assert audits[model][view]['locally_cited']['current']==int(a[:,0].sum())
    full={cfg:{str(seed):f'{cfg}_s{seed}' for seed in r.SEEDS} for cfg in r.CONFIGS}
    means={cfg:{} for cfg in r.CONFIGS};mean_audits={cfg:{} for cfg in r.CONFIGS}
    single_cis={};mean_cis={};hier_cis={};draw_checks={}
    for view in r.VIEWS:
        n=tables['base'][view]['n'];qix=np.random.default_rng(20261010).integers(0,n,(20000,n)) if n else np.empty((20000,0),int)
        six=np.random.default_rng(20261011).integers(0,3,(20000,3))
        aa={m:arrays[m][view] for m in models};ob={m:observed(v) for m,v in aa.items()};dr={m:bootstrap(v,qix) for m,v in aa.items()}
        for m,b in r.single_pairs():single_cis[m+'__minus__'+b+'__'+view]=interval(ob[m]-ob[b],dr[m]-dr[b],'paired_questions_single_model')
        stacks={cfg:np.stack([aa[full[cfg][str(seed)]] for seed in r.SEEDS]) for cfg in r.CONFIGS}
        stacks.update({b:np.repeat(aa[b][None],3,axis=0) for b in ['base','base_block','anchor']})
        mo={cfg:observed(v.mean(0)) for cfg,v in stacks.items()}
        md={cfg:bootstrap(v.mean(0),qix) for cfg,v in stacks.items()}
        hd={cfg:bootstrap(v,qix,six) for cfg,v in stacks.items()}
        for cfg in r.CONFIGS:
            measurements=np.stack([ob[full[cfg][str(seed)]] for seed in r.SEEDS])
            means[cfg][view]={f:mean_record(measurements[:,i]) for i,f in enumerate(FIELDS)}
            assert np.allclose(np.mean(measurements,0),mo[cfg],rtol=0,atol=1e-12,equal_nan=True)
            mean_audits[cfg][view]={}
            for mode in ['name_only','locally_cited','Q_decomposition']:
                mean_audits[cfg][view][mode]={field:mean_record([audits[full[cfg][str(seed)]][view][mode][field] for seed in r.SEEDS]) for field in audits[full[cfg][str(r.SEEDS[0])]][view][mode]}
            frequencies=[]
            for k,row in sorted(byid.items()):
                if view!='full311' and row['eval_split']!=view:continue
                bn=members(row,predictions['base'][k]);bc=members(row,predictions['base'][k],True)
                counts={name:{} for name in ['new_name_with_supported_citation','old_name_citation_repaired','lost_supported_group']}
                for seed in r.SEEDS:
                    c=members(row,predictions[full[cfg][str(seed)]][k],True)
                    for name,ss in [('new_name_with_supported_citation',c-bn),('old_name_citation_repaired',(c&bn)-bc),('lost_supported_group',bc-c)]:
                        for i in ss:counts[name][str(i)]=counts[name].get(str(i),0)+1
                frequencies.append({'id':k,'seeds':3,'counts_0_to_3':counts,'edges':row['edges']})
            r.rows(folder/'ledgers'/(cfg+'__seed_frequency__'+view+'.jsonl'),frequencies)
        for m,b in r.mean_pairs():
            key=m+'__minus__'+b+'__'+view
            mean_cis[key]=interval(mo[m]-mo[b],md[m]-md[b],'paired_questions_mean_of_fixed_three_seeds')
            hier_cis[key]=interval(mo[m]-mo[b],hd[m]-hd[b],'paired_seed_and_question_two_level_only_three_seed_clusters')
        draw_checks[view]={'questions':n,'Q_effective_questions':tables['base'][view]['Q_effective_questions'],
            'Q_denominator':tables['base'][view]['gold_edges'],'zero_Q_denominator_draws':int(np.isnan(dr['base'][:,0]).sum())}
    summary={'models':models,'seeds':r.SEEDS,'configs':r.CONFIGS,'views':r.VIEWS,'tables':tables,'name_audits':audits,
        'full_config_seed_keys':full,'three_seed_mean_SD':means,'three_seed_mean_audits':mean_audits,
        'paired_bootstrap':single_cis,'three_seed_mean_paired_bootstrap':mean_cis,'two_level_paired_bootstrap':hier_cis,
        'bootstrap_denominators':draw_checks,'new_training_runs':0,'new_predictions':len(models)*len(inputs),
        'no_model_selection':True,'no_multiple_comparison_adjustment':True,'single_CI_count':len(single_cis),
        'mean_CI_count':len(mean_cis),'hierarchical_CI_count':len(hier_cis)}
    summary=clean(summary);save(folder/'summary.json',summary)
    return summary


def name(key):
    if key in LABEL:return LABEL[key]
    cfg,seed=key.split('_s');return LABEL[cfg]+' Seed '+seed
def number(v,scale=1):return 'NA' if v is None else f'{v*scale:.2f}'
def cell(rec,scale=1):return number(rec['mean'],scale)+' ± '+number(rec['SD_sample_ddof1'],scale)
def ci(rec,f):
    v=rec[f];scale=1 if f=='mean_tokens' else 100
    def fmt(x):return 'NA' if x is None else f'{x*scale:+.2f}'
    return fmt(v['delta'])+' ['+fmt(v['ci95'][0])+', '+fmt(v['ci95'][1])+']'


def report(s,preflight):
    lines=['# EQUAL-RAG：311题预注册一次性最终评测报告','',
        '## 0. 完成情况与解释边界','',
        '本报告覆盖既有311题封存集：3基线＋3配置×3训练Seed，共12系统、3732条一次性生成。零新训练、零超参数搜索、零checkpoint挑选；所有主方法为未合并LoRA标准贪心自由生成。只有Base＋防重探针使用既有解码规则。',
        '内部预注册协议、权重与代码SHA在读取封存题内容及生成之前固化；输入与切片清单、提示token在生成之前冻结。不冒称第三方公开注册或底座全预训练历史无污染。最终结果无论正负均完整披露，不由分数触发后续模型调整。',
        '三Seed平均是逐题充分统计量的平均，不是合并模型、答案并集或输出投票。SD为跨训练Seed的样本标准差；主CI固定3个Seed、重采样问题，补充CI配对重采样Seed与问题。两种CI均20,000次，95%百分位区间。',
        '未指定独占主配置或单一主终点，全部区间为逐比较、未作多重比较校正；不能用最有利的一条区间证明整个家族显著。只有3训练Seed，双层Bootstrap并不消除小Seed数局限。',
        'Q沿用开发计分函数，是检索文本中词面支持、列表单位内引用匹配的组级micro召回，不是语义蕴涵评分。Full F1对完整gold逐题macro平均；其余所有指标也保留零支持题。词面多答案子集不等于人工验证的可答集。','',
        '### 实际结果摘要','']
    for cfg in r.CONFIGS:
        m=s['three_seed_mean_SD'][cfg]['full311'];rec=s['three_seed_mean_paired_bootstrap'][cfg+'__minus__base__full311']
        lines.append('- '+LABEL[cfg]+'：全集Q '+cell(m['grounded_quality'],100)+'%，Full F1 '+cell(m['answer_f1'],100)+'%，Token '+cell(m['mean_tokens'])+'；相对Base ΔQ '+ci(rec,'grounded_quality')+'、ΔF1 '+ci(rec,'answer_f1')+'（百分点）。')
    lines+=['','## 1. 预设切片、分母与血缘核验','',
        '| 视图 | 题数 | Q有效题数 | Q支持组分母 | 零支持题数 | 最大单题支持组数 |','|---|---:|---:|---:|---:|---:|']
    for view in r.VIEWS:
        v=preflight['strata'][view];lines.append(f"| {VIEWS[view]} | {v['questions']} | {v['Q_effective_questions']} | {v['supported_group_denominator']} | {v['zero_support_questions']} | {v['largest_question_groups']} |")
    lines+=['','两子集按≥2／0–1支持组划分，互斥且覆盖全集，不按输出选题。历史Raw24包含1道≥2支持组问题，故这里按预注册阈值对齐，不声称复现历史Raw24抽样。',
        '与已追溯350训练题、160开发题、最近69开发题的ID及两种规范化问题重合均为0。此为精确匹配核验，不认证语义近重复、预训练记忆或所有未知历史源。',
        f"最大提示长度 {preflight['prompt_tokens_max']} tokens；所有题无提示截断。推理数据投影仅包含ID、提示及提示token，gold、支持组和切片不传给解码器。",'',
        '## 2. 主表：基线与三Seed Mean ± SD','', 'Q、F1、重复率和真触顶率用百分数；Token为绝对数。Base/Anchor/探针各仅一次生成，不伪造Seed SD。','']
    for view in r.VIEWS:
        lines+=['### '+VIEWS[view],'','| 系统 | Q % | Full F1 % | 平均Token | 重复率 % | 真触顶率 % |','|---|---:|---:|---:|---:|---:|']
        for b in ['base','base_block','anchor']:
            t=s['tables'][b][view];lines.append('| '+LABEL[b]+' | '+' | '.join(number(t[f],1 if f=='mean_tokens' else 100) for f in FIELDS)+' |')
        for cfg in r.CONFIGS:
            t=s['three_seed_mean_SD'][cfg][view];lines.append('| '+LABEL[cfg]+' | '+' | '.join(cell(t[f],1 if f=='mean_tokens' else 100) for f in FIELDS)+' |')
        lines.append('')
    lines+=['## 3. 每个训练Seed明细','']
    for view in r.VIEWS:
        lines+=['### '+VIEWS[view],'','| 系统 | Q % | Full F1 % | 平均Token | 重复率 % | 真触顶率 % |','|---|---:|---:|---:|---:|---:|']
        for model in s['models'][3:]:
            t=s['tables'][model][view];lines.append('| '+name(model)+' | '+' | '.join(number(t[f],1 if f=='mean_tokens' else 100) for f in FIELDS)+' |')
        lines.append('')
    lines+=['## 4. 相对Base的名字与支持引用组得失','',
        '真新名字＋支持引用＝当前命中支持引用、Base没有输出该名字；原名字引用修复＝Base已输出名字但未命中支持引用，当前补成。二者合计等于支持引用组新增。名字不带引用的得失单列，不混同Q。','']
    for view in r.VIEWS:
        lines+=['### '+VIEWS[view],'','| 系统 | 名字新增/丢失/净增 | 引用组新增/丢失/净增 | 真新名字＋引用 | 原名字引用修复 | 原支持名字消失 | 名字保留但引用丢失 |','|---|---|---|---:|---:|---:|---:|']
        for model in s['models']:
            a=s['name_audits'][model][view];d=a['Q_decomposition']
            nums=['/'.join(str(a[m][f]) for f in ['added','lost','net']) for m in ['name_only','locally_cited']]
            lines.append('| '+name(model)+' | '+' | '.join(nums+[str(d[f]) for f in ['new_name_with_supported_citation','old_name_citation_repaired','original_supported_name_missing','name_retained_citation_lost']])+' |')
        lines+=['','| 三Seed配置 | 引用组新增 | 引用组丢失 | 引用组净增 | 真新名字＋引用 | 原名字引用修复 | 名字净增 |','|---|---:|---:|---:|---:|---:|---:|']
        for cfg in r.CONFIGS:
            a=s['three_seed_mean_audits'][cfg][view]
            vals=[cell(a['locally_cited'][f]) for f in ['added','lost','net']]+[cell(a['Q_decomposition'][f]) for f in ['new_name_with_supported_citation','old_name_citation_repaired']]+[cell(a['name_only']['net'])]
            lines.append('| '+LABEL[cfg]+' | '+' | '.join(vals)+' |')
        lines.append('')
    for heading,group in [('5. 三Seed平均的问题级配对区间','three_seed_mean_paired_bootstrap'),('6. 双层Seed＋问题配对区间','two_level_paired_bootstrap'),('7. 逐Seed与基线全部问题级配对区间','paired_bootstrap')]:
        lines+=['## '+heading,'','差值为前者−后者；Q/F1/重复/触顶为百分点，Token为绝对数。','']
        for view in r.VIEWS:
            lines+=['### '+VIEWS[view],'','| 比较 | ΔQ [95% CI] | ΔFull F1 [95% CI] | ΔToken [95% CI] | Δ重复 [95% CI] | Δ真触顶 [95% CI] |','|---|---|---|---|---|---|']
            for key,rec in s[group].items():
                m,_,b,v=key.split('__')
                if v==view:lines.append('| '+name(m)+' − '+name(b)+' | '+' | '.join(ci(rec,f) for f in FIELDS)+' |')
            den=s['bootstrap_denominators'][view]
            lines+=['',f"问题级Q分母为0的抽样次数：{den['zero_Q_denominator_draws']}/20,000；仅这些抽样的Q不定义，其他指标使用全部抽样。",'']
    lines+=['## 8. 可报告结论与限制','',
        '- 本轮为当前家族被冻结后、既有封存集上的一次性确认性评测；不挑配置或checkpoint、不回流训练、不因切片分数重划分数据。',
        '- 正向点估计、单个未校正CI不跨零、家族整体显著、跨模型可推广是不同主张；不得互相替代。所有比较均完整给出，未作多重比较校正。',
        '- β=0是纯CE框架，不能改名为Pair；AOnly与CE若没有明确差异，应如实写成权衡或未检出额外优势，而不是证明等效。',
        '- Q与Full F1不同，Q高可能来自引用修复或少数密集题，不能全归因于真新实体召回；需结合分解表和逐题审计。',
        '- 新增/丢失均是已知、词面支持的gold组。没有语义蕴涵判定或穷尽式人工评审，不能声称所有引用均真正支持问题关系或无幻觉。',
        '- 真触顶指标来自实际token和原生EOS；重复定义沿用逗号列表单位，可能漏掉非列表循环和别名重复。区间跨0不证明长度无膨胀。',
        '- 该311为ALCE QAMPARI来源的项目内冻结切分，不是官方完整测试集；底座整个预训练史、所有不可追溯历史源及语义近重复无法由精确ID核验排除。',
        '- 此报告不自动启动任何新实验、训练、参数搜索、模型替换或第二轮封存评测。','',
        '## 9. 冻结与归档','',
        f"封存原文件SHA256：`{r.HELD_SHA}`。",
        'protocol.json、freeze.json、preflight.json提供注册时间、代码与权重SHA、切片ID及输入SHA；predictions/提供3732条原始预测、每题唯一尝试记录、提示/生成token与EOS遥测；analysis/提供计分、逐题组审计、跨Seed频次和全部207组CI。',
        '权重保持原归档路径，不重复打包大权重；归档保存每个权重文件SHA及各次推理未合并/无训练验证。最终完整性核验、SHA清单和独立本地复算记录另附。',
        f'运行目录：`{r.RUN}`。','']
    return '\n'.join(lines)


def finish():
    assert r.js(r.RUN/'status.json')['state']=='ALL_GENERATIONS_COMPLETE'
    protocol=r.check_freeze();inputs=r.read(r.RUN/'data/eval311.jsonl');pred={}
    for model in protocol['systems']:
        folder=r.RUN/'predictions'/model;status=r.js(folder/'status.json');manifest=r.js(folder/'manifest.json')
        assert status['state']=='COMPLETE' and manifest['trainable_parameters']==0 and not manifest['merge_called'] and not manifest['training_called']
        attempts=r.read(folder/'attempts.jsonl');ps=r.read(folder/'predictions.jsonl')
        assert len(attempts)==len(ps)==len({p['id'] for p in ps})==311
        assert [v['id'] for v in attempts]==[v['id'] for v in ps]==[v['id'] for v in inputs]
        assert all(v['attempt']==1 for v in attempts)
        assert r.sha(folder/'predictions.jsonl')==status['predictions_sha256']
        pred[model]={v['id']:v for v in ps}
    summary=statistics(inputs,pred,r.RUN/'analysis')
    assert [summary[k] for k in ['single_CI_count','mean_CI_count','hierarchical_CI_count']]==[117,45,45]
    (r.RUN/'EQUAL_HELDOUT311_FINAL_RESULTS_20261015.md').write_text(report(summary,r.js(r.RUN/'preflight.json')),encoding='utf-8')
    unchanged=[]
    for item in protocol['base_files']+[x for spec in protocol['systems'].values() for x in spec['files']]:
        assert r.sha(item['path'])==item['sha256'],item['path'];unchanged.append(item)
    save(r.RUN/'completion_verification.json',{'state':'PASS','utc':r.now(),'models':12,'predictions':3732,'unique_attempts':3732,
        'new_training_runs':0,'all_frozen_model_files_unchanged':True,'checked_weight_and_config_files':len(unchanged),
        'single_CI_count':117,'mean_CI_count':45,'hierarchical_CI_count':45,'all_generations_before_any_quality_analysis':True,
        'original_holdout_sha256':r.sha(r.HELD),'freeze_sha256':r.sha(r.RUN/'freeze.json')})
    r.dump(r.RUN/'status.json',{'state':'COMPLETE','utc':r.now(),'models':12,'predictions':3732,'new_training_runs':0})
    archive()


def archive():
    dest=r.ROOT/'runs/qwen_heldout311_closeout_20261015';dest.mkdir(exist_ok=False)
    files=[p for p in sorted(r.RUN.rglob('*')) if p.is_file()]
    manifest={'run/'+str(p.relative_to(r.RUN)):{'bytes':p.stat().st_size,'sha256':r.sha(p)} for p in files}
    save(dest/'manifest.json',manifest)
    target=dest/'qwen_heldout311_20261015.tar.gz'
    with tarfile.open(target,'w:gz') as tf:
        tf.add(dest/'manifest.json',arcname='manifest.json',recursive=False)
        for p in files:tf.add(p,arcname='run/'+str(p.relative_to(r.RUN)),recursive=False)
    save(dest/'archive_summary.json',{'archive':str(target),'sha256':r.sha(target),'bytes':target.stat().st_size,'files':len(files)+1,
        'report_sha256':r.sha(r.RUN/'EQUAL_HELDOUT311_FINAL_RESULTS_20261015.md'),'predictions':3732,'new_training_runs':0})
    print(json.dumps({'state':'ARCHIVED','archive':str(target),'sha256':r.sha(target)}),flush=True)


if __name__=='__main__':finish()

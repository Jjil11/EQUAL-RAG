"""Full Chinese paper-data report and immutable, hash-verified artifact bundle."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import analyze_tail_open_suite_20261015 as a

q=a.q;ROOT=a.ROOT;SOURCE=a.SOURCE
OUT=q.ROOT/'runs/qwen_tail_open_suite_closeout_20261015'
LABEL=a.LABEL


def interval_cells(record):
    cells=[]
    for field in a.FIELDS:
        x=record[field];scale=1 if field=='mean_tokens' else 100
        cells.append(f"{x['delta']*scale:+.2f} [{x['ci95'][0]*scale:+.2f}, {x['ci95'][1]*scale:+.2f}]")
    return cells


def metric_tables(lines, summary, models):
    for split,title in [('multi45','45题多答案子集'),('raw24','Raw24')]:
        lines += ['### '+title,'','| 系统 | Q % | Full F1 % | 平均Token | 重复率 % | 真触顶率 % |','|---|---:|---:|---:|---:|---:|']
        for m in models:
            r=summary['tables'][m][split]
            lines.append(f"| {LABEL[m]} | {100*r['grounded_quality']:.2f} | {100*r['answer_f1']:.2f} | {r['mean_tokens']:.2f} | {100*r['repetition_rate']:.2f} | {100*r['true_cap_hit_rate']:.2f} |")
        lines.append('')


def audit_table(lines, summary, models, split='multi45'):
    identity_header='上轮新增17组仍保留 | 上轮丢失17组已恢复' if split=='multi45' else '历史新增仍保留 | 历史丢失已恢复'
    lines += ['| 系统 | 名字新增 | 名字丢失 | 名字净增 | 带支持引用组新增 | 带支持引用组丢失 | 引用组净增 | 真新名字＋支持引用 | 原名字引用修复 | '+identity_header+' |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        r=summary['name_audits'][m][split];n=r['name_only'];g=r['locally_cited'];d=r['Q_decomposition']
        lines.append(f"| {LABEL[m]} | {n['added']} | {n['lost']} | {n['net']:+d} | {g['added']} | {g['lost']} | {g['net']:+d} | {d['new_name_with_supported_citation']} | {d['old_name_citation_repaired']} | {g['old_added_retained']} | {g['old_lost_recovered']} |")
    lines.append('')


def training_table(lines, summary, models):
    lines += ['| 配置 | Prefix Top-1 % | 整条保留 /120 | A /21 | C1 /21 | C2 /20 | A完整已知链 /21 | A训练题前缀保留 /14 | A训练题引用组新增/丢失 | A训练题平均Token |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for m in models:
        p=summary['profiles'][m];f=p['fidelity'];d=p['diagnostic'];t=p['training_A14'];g=t['locally_cited']
        lines.append(f"| {LABEL[m]} | {100*f['prefix_top1_fidelity']:.3f} | {f['fully_preserved_trajectories']} | {d['by_type']['A']['valid_type_success']} | {d['C1']['success']} | {d['by_type']['C']['valid_type_success']} | {d['by_type']['A']['complete_chain_success']} | {t['full_prefix_preserved']} | {g['added']}/{g['lost']} | {t['mean_tokens']:.2f} |")
    lines.append('')


def report(summary):
    p=summary['name_audits']['pair60']['multi45']['locally_cited']
    ce=summary['name_audits']['ce60']['multi45']['locally_cited']
    gp=summary['name_audits']['guard_pair']['multi45']['locally_cited']
    gpf=summary['profiles']['guard_pair']['fidelity']['fully_preserved_trajectories']
    scales=['scale070','scale085','scale100','scale115']
    guards=['guard_ce','guard_pair']
    lines=['# EQUAL-RAG：取消训练门控后的完整评测、静态合并扫描与Max-Guard对照','',
        '日期：2026-10-15；底座：Qwen3-14B；所有主方法均为标准greedy自由生成。','',
        '## 0. 先看结论与口径','',
        f"现有60步Pair在45题上：带支持引用组新增{p['added']}、丢失{p['lost']}、净增{p['net']:+d}。同预算CE新增{ce['added']}、丢失{ce['lost']}、净增{ce['net']:+d}。训练侧Pair优于CE的局部命中排序没有直接转化为本轮开发集排序。",
        f"Max-Guard Pair融合后：整条保留{gpf}/120，45题带支持引用组新增{gp['added']}、丢失{gp['lost']}、净增{gp['net']:+d}。是否同时达到前缀≥110且净增>0：{'是' if gpf>=110 and gp['net']>0 else '否'}。这不等于保住上轮新增17组，也不自动代表Full F1或整体质量提升；须同时看下文强CE和同精度融合控制。",
        '下文完整报告所有预定配置，不因A、C1、C2或前缀一致率未达标而屏蔽任何已成功运行的模型。数据/代码错误和OOM仍应修复并如实记录，不能为“必须输出”伪造结果。',
        '45题与Raw24均为已反复使用的开发视图。静态λ的最佳点还会利用这些视图选择，因此不是独立盲测结论；单随机种子、未做多重比较校正的区间也不能替代后续确认。311题封存集读取及生成均为0。','',
        '## 1. 实际执行协议','',
        '- 第一步直接加载上轮已冻结的CE60与Pair60权重，绕过原训练门控，不修改历史失败记录、不重训或挑另一个checkpoint。前三个基线复用同一输入token的既有缓存。',
        '- 所有69题输入逐token与Base缓存核对；关闭thinking，do_sample=False，max_new_tokens=256。主方法无防重探针、解码路由、打分重排、答案截断或外部停止规则。仅原始Qwen＋防重探针这一对照保留其规则。',
        '- 词面Q是支持引用组的micro召回代理；45题分母176。Full F1是完整gold答案的逐题F1均值。引用词面支持不等于语义关系蕴涵证明。',
        '- Raw24仅4题、8个已知支持组贡献Q，另外20题无已知词面支持不等于语义不可答。不能把Raw24的Q当成24题全部答案正确率。',
        '- 重复率沿用项目的规范化逗号列表口径，可能漏掉缺逗号的循环和别名重复；真触顶按实际生成256 token且未输出EOS计算。平均Token包含终止token。',
        '- 静态扫描为λ∈{0.70,0.85,1.00,1.15}。仅乘一次LoRA增量：W融合=W原始+λ×(alpha/r)×BA，不能同时缩放A、B变成λ²。',
        '- 所有静态配置真实执行scale_layer和safe_merge，融合后的LoRA层数为0。只在内存中一次性融合，不导出多份28GB底座；生成时没有额外adapter分支或规则。未实测时延/吞吐，不能宣称实测加速。',
        '- BF16融合存在舍入差异，λ=1合并模型不保证与未合并LoRA逐token相同。因此另保留CE60 λ=1融合控制；Guard应优先与各自同样融合的60步控制比较，不能把融合差异归因于Max-Guard。',
        '- 另做了一个CPU数值反例核验（不是新增Qwen评测配置）：底座权重[1,−1]、输入[1,1]、LoRA增量[0.002,0]，未融合输出0.0019989013671875，BF16融合后增量舍入掉、输出变为0。它证明“不保证融合等效”确实可能发生，但不证明实测Q变化全部由这一机制造成。详见precision_toy.json。',
        '- Max-Guard两臂均从官方Base重新新建LoRA，固定60步、梯度累积4；各240次监督访问＋240次辅助原生保留访问。与原60步的初始化、标签、保留位点、访问顺序、优化器设置一致，不从旧60步继续训练，也不在测试集上选停止步。',
        '- 上轮93/120的准确含义是27条轨迹“至少有一处”翻转，并非各恰好一个token：CE共38个保留位点翻转，Pair共40个。保留均值项并新增max项，也避免只盯最差一处而完全忽略其他翻转。',
        '- 原实现的保序均值是“仅翻转位置平均”，不是所有token平均。本轮保留该项，并新增每轨迹max项：', '',
        '`L保留 = 10 KL(Base||Student) + 2 mean_changed[relu(0.5−margin)] + 2 max_retained[relu(0.3−margin)]`','',
        '`margin = logP(Base_top1) − max_other logP(other)`','',
        'max项覆盖全部保留位点，包括argmax仍相同但间隔不足0.3的位点；原均值项仍仅在翻转时触发。A/正跨度/首新实体正CE/C1终端CE以及Pair结构化Hinge均沿用上轮，不增加跨样本普通名字的负Hinge。',
        'A仍为21标签、14独立原生前缀；C1为21个A路径内的收尾标签，C2为20条原生停止保持，B=0。保留池为120条、3174个保留位点。训练与69题按ID及规范化问题文本均无重叠。','',
        '## 2. 第一步：原有60步模型完整五系统结果','']
    metric_tables(lines,summary,['base','base_block','anchor','ce60','pair60'])
    lines += ['### 45题名字组得失审计（相对原始Base）','']
    audit_table(lines,summary,['ce60','pair60'])
    lines += ['“带引用组新增”包括原有名字补对引用，不能全部写成新补出的实体。上轮17组的身份追踪单独列出，不能把这轮新增总数误称为保住了上轮全部17组。','',
        '### 第一步20,000次问题级配对Bootstrap','',
        '差值为前者−后者。Q、F1、重复、真触顶使用百分点，Token使用绝对数；95%百分位区间。','',
        '| 数据集 | 比较 | ΔQ [95% CI] | ΔFull F1 [95% CI] | ΔToken [95% CI] | Δ重复 [95% CI] | Δ真触顶 [95% CI] |',
        '|---|---|---|---|---|---|---|']
    for split in ['multi45','raw24']:
        for m,b in [('pair60','base'),('pair60','ce60'),('ce60','base')]:
            r=summary['paired_bootstrap'][m+'__minus__'+b+'__'+split]
            lines.append('| '+split+' | '+LABEL[m]+' − '+LABEL[b]+' | '+' | '.join(interval_cells(r))+' |')
    lines += ['','## 3. 第二步A：静态LoRA系数扫描','',
        '所有λ都完整运行120条前缀核验、A/C1/C2条件诊断、14道A训练题从完整提示自由生成，以及45题／Raw24评测。14题结果只作训练诊断，不作为泛化证据。','']
    training_table(lines,summary,scales)
    metric_tables(lines,summary,scales)
    lines += ['### 静态扫描45题名字组得失','']
    audit_table(lines,summary,scales)
    balance=summary['static_scan_balance']
    best=balance['development_selected_under110_constraint']
    lines += ['### 预先固定的平衡点规则','',
        '先限制整条保留≥110/120，在可行集合中最大化45题带引用组净增；并列时先取Raw24 token较少，再取λ离1较近。若没有配置达到110，不强造单个“达标最佳”，只报告保留轨迹数／45题净增的Pareto折中。',
        '达到110的配置：'+('、'.join(LABEL[m] for m in balance['threshold110_feasible']) or '无')+'。',
        '该规则选中的开发点：'+(LABEL[best] if best else '无达标点')+'。',
        '同时满足“整条保留≥110且45题引用组净增>0”：'+('是' if any(v['whole_prefix']>=110 and v['cited_net45']>0 for v in balance['values'].values()) else '否')+'。仅过保留门槛而净增为0或负数，不能称为端到端目标达成。',
        '保留率／净增的Pareto前沿：'+'、'.join(LABEL[m] for m in balance['pareto_retention_vs_net45'])+'。',
        '这是开发集选择，不是经独立数据确认的最优超参数；下文照样保留全部配置及区间。','',
        '## 4. 第二步B：固定60步Max-Violation Prefix Guard','',
        '下列四个模型均真实融合后生成。CE／Pair的λ=1融合控制用于区分guard变化与BF16融合舍入变化。不存在训练质量拦截或中途早停。','']
    training_table(lines,summary,['ce60_merged','scale100','guard_ce','guard_pair'])
    metric_tables(lines,summary,['ce60_merged','scale100','guard_ce','guard_pair'])
    lines += ['### Guard45题名字组得失','']
    audit_table(lines,summary,['ce60_merged','scale100','guard_ce','guard_pair'])
    lines += ['## 5. 全部配置45题及Raw24得失总审计','',
        '同一题同一规范化组只计一次。主引用组库存与Q完全一致。逐题名字、别名、支持文档、旧新增保留、旧丢失恢复及分解均保存在analysis_final/ledgers。','',
        '### 45题','']
    audit_table(lines,summary,[m for m in summary['models'] if m!='base'])
    lines += ['### Raw24（此处“旧17组”列不适用，只是该视图的历史得失身份追踪）','']
    audit_table(lines,summary,[m for m in summary['models'] if m!='base'],'raw24')
    lines += ['## 6. 所有配对区间（20,000次）','',
        '所有配置相对Base；另含Pair60−CE60、缩放−未合并Pair60、缩放−融合λ=1、Guard Pair−Guard CE及各Guard与同类融合控制。固定seed20261010，逐题配对重采样，未做多重比较或超参数选择修正。区间跨0不证明优势，也不证明等效；[0,0]也不是非劣检验。','']
    for split in ['multi45','raw24']:
        lines += ['### '+split,'','| 比较 | ΔQ [95% CI] | ΔFull F1 [95% CI] | ΔToken [95% CI] | Δ重复 [95% CI] | Δ真触顶 [95% CI] |','|---|---|---|---|---|---|']
        for key,r in summary['paired_bootstrap'].items():
            m,_,b,s=key.split('__')
            if s!=split:continue
            lines.append('| '+LABEL[m]+' − '+LABEL[b]+' | '+' | '.join(interval_cells(r))+' |')
        r=summary['paired_bootstrap']['pair60__minus__base__'+split]
        lines += ['',f"Q分母为0的重采样为{r['zero_Q_denominator_draws']}次，仅从Q区间剔除；其他指标仍保留全部20,000次。",'']
    influence=json.loads((ROOT/'question_influence45.json').read_text())
    lines += ['## 7. 逐题贡献与论文解释','',
        '以下敏感性检查没有删题，也不用于更换主评分。它说明micro组召回的净增是否集中在少数多答案题；“去掉最大增益题”只是透明披露，不是另一套主结果。','',
        '| 系统 | 净增为正的题 | 净增为负的题 | 最大单题净增 | 去掉最大增益题后的引用组净增 |',
        '|---|---:|---:|---:|---:|']
    for m in [m for m in summary['models'] if m not in ['base','base_block','anchor']]:
        x=influence[m]
        lines.append(f"| {LABEL[m]} | {x['positive_net_questions']} | {x['negative_net_questions']} | {x['largest_question_gain']} | {x['net_without_largest_gain_question']:+d} |")
    lines += ['',
        '- 应报告本轮真实强CE控制。不能因Pair在训练A/C1命中更高，就称它在未见题更好；也不能把CE结果改名为Pair或略去不利对照。',
        '- 应分开报告：真新名字补齐、已有名字引用修复、旧名字丢失、引用丢失。Q升高不能直接改写为同数量的新实体被召回。',
        '- 若某配置只降低丢失，却同时削弱新增，必须按净变化汇报；110/120训练前缀保持也不是45题无损保证。',
        '- max只惩罚最差单点，是软训练目标，不是数学上锁死全部前缀。训练teacher-forcing的一致性与未见题从头生成仍须区分。',
        '- 本轮同时有监督设计、停止校准、KL/保序及融合等因素，不能单独证明上轮损失17组完全由旧C导致。',
        '- 45题和Raw24已用于反复开发，λ选择也使用同一视图；本轮结果应标为开发集、单seed、探索性证据。正式论文若主张泛化或显著优势，仍需预注册配置后的独立评估、多seed与分组件消融。',
        '- 词面gold集合不穷尽所有真实正确答案；C2是原生句尾保持，不是语义证据穷尽证明。引用代理也不等于完整事实正确性。','',
        '## 8. 复现与文件核验','',
        '新增7项单元测试通过：max项数值/梯度、低间隔预防项、原翻转均值保留、λ仅乘一次与真实融合、向量化Bootstrap和历史实现一致。最初测试夹具缺少汇总字段导致的一次测试错误记录也保留，修正后全通过；不涉及模型结果筛除。',
        '现有60步权重未修改，历史门控失败包未覆盖。本轮全部9个新评测配置共621条新评测预测，另有7配置×14道训练题的完整提示诊断。完整预测、标签、统计、日志、参数SHA和代码一并归档，不包含底座、LoRA权重和参考分布张量。',
        '复制的历史protocol/audit中有原轮门控及“此前无开发生成”等继承字段，只描述原轮历史，不描述本轮。当前固定60步、无门控和已复用开发集的实际执行，以本轮代码、新协议字段及inherited_protocol_scope_notes.json为准；原始记录不改写。',
        f'运行目录：`{ROOT}`。封存311题读取及生成均为0。','']
    target=ROOT/'EQUAL_OPEN_EVAL_SCAN_GUARD_RESULTS_20261015.md'
    assert not target.exists()
    target.write_text('\n'.join(lines),encoding='utf-8')
    return target


def verify(summary):
    tests=(q.ROOT/'logs/qwen_tail_open_unit_tests_v2_20261015.log').read_text()
    assert 'Ran 7 tests' in tests and tests.rstrip().endswith('OK')
    originals={m:json.loads((SOURCE/m/'manifest.json').read_text()) for m in ['base_ce','base_pair']}
    guards={m:json.loads((ROOT/'guard60'/m/'manifest.json').read_text()) for m in ['base_ce','base_pair']}
    for m in guards:
        assert guards[m]['updates']==60 and guards[m]['supervised_visits']==240
        for key in ['initial_trainable_weights_sha256','data_sha256','retention_sha256','order_sha256','aux_order_sha256']:
            assert guards[m][key]==originals[m][key],key
        assert q.sha(SOURCE/m/'adapter/adapter_model.safetensors')==originals[m]['adapter_sha256']
    for name in ['compiled/records.jsonl','compiled/native_retention.jsonl','eval69.jsonl']:
        assert q.sha(ROOT/'guard60'/name)==q.sha(SOURCE/name)
    for m in summary['profiles']:
        meta=json.loads((ROOT/'profiles'/m/'manifest.json').read_text())
        assert meta['LoRA_layers_after_merge']==0 and meta['merged_once'] and not meta['per_token_LoRA_or_rule_overhead']
    assert len(summary['models'])==12 and len(summary['profiles'])==7
    assert len(summary['paired_bootstrap'])==48
    toy=json.loads((ROOT/'precision_toy.json').read_text())
    assert toy['status']=='PASS' and toy['toy_not_Qwen_eval']
    assert toy['unmerged_output']!=0 and toy['BF16_merged_output']==0
    q.dump(ROOT/'inherited_protocol_scope_notes.json',{
        'guard_protocol_inherited_gate_fields_are_inactive':['A_gate_min','prefix_fidelity_min','gate_every','independent_first_gate_stopping_both_arms'],
        'actual_guard_execution':'fixed60_no_break_no_early_stop_no_quality_filter',
        'no_prior_dev_generation_inherited_from_source_V3_not_current_suite':True,
        'actual_current_suite_reuses_dev_and_executes_stage1_before_guard':True,
        'compiled_audit_byte_identity_inherited_V2_scope_not_V3_metadata_bytes':True,
        'labels_and_positive_token_paths_unchanged':True,'raw_historical_fields_preserved_not_rewritten':True})
    q.dump(ROOT/'completed_verification.json',{'status':'PASS','unit_tests':7,
        'all_12_systems_complete':True,'all_7_merged_profiles_complete':True,'new_eval_predictions':621,
        'all_inputs_matched_by_tokens':True,'guard_same_init_labels_order_and_60_step_budget':True,
        'no_training_gate_or_metric_filter':True,'historical_source_weights_unchanged':True,
        'all_merged_LoRA_layers':0,'BF16_numerical_counterexample_verified':True,
        'all_bootstraps_20000':True,'paired_comparisons':48,
        'sealed311_reads':0})


def question_influence(summary):
    out={}
    for m in summary['models']:
        if m=='base':continue
        rows=q.read(ROOT/'analysis_final/ledgers'/(m+'__multi45.jsonl'))
        net=[(len(r['locally_cited']['added'])-len(r['locally_cited']['lost']),r['id']) for r in rows]
        biggest=max(net)
        total=sum(v for v,_ in net)
        assert total==summary['name_audits'][m]['multi45']['locally_cited']['net']
        out[m]={'positive_net_questions':sum(v>0 for v,_ in net),
            'negative_net_questions':sum(v<0 for v,_ in net),'zero_net_questions':sum(v==0 for v,_ in net),
            'largest_question_gain':biggest[0],'largest_gain_question_id':biggest[1],
            'total_net':total,'net_without_largest_gain_question':total-biggest[0],
            'not_used_to_delete_questions_or_select_checkpoints':True}
    q.dump(ROOT/'question_influence45.json',out)


def main():
    summary=json.loads((ROOT/'analysis_final/summary.json').read_text())
    verify(summary)
    question_influence(summary)
    target=report(summary)
    q.dump(ROOT/'environment.json',{'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','numpy']},
        'Base_config_sha256':q.sha(Path(q.MODEL)/'config.json'),'tokenizer_sha256':q.sha(Path(q.MODEL)/'tokenizer.json')})
    shutil.copy2(c_previous(),ROOT/'historical100_Pair_predictions69.jsonl')
    q.dump(ROOT/'status.json',{'state':'COMPLETE','new_eval_predictions':621,'training_gate_used':False,
        'sealed311_reads':0,'all_declared_configurations_reported':True})
    assert not OUT.exists();OUT.mkdir()
    selected=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.md']:
            selected.append((p,'run/'+p.relative_to(ROOT).as_posix()))
    names=['qwen_tail_open_eval_20261015.py','qwen_tail_guard_scan_20261015.py','analyze_tail_open_suite_20261015.py',
        'closeout_tail_open_suite_20261015.py','test_tail_open_suite_20261015.py','status_tail_open_suite_20261015.py',
        'check_tail_bf16_merge_precision_20261015.py',
        'qwen_tail_fidelity_20261015.py','qwen_tail_fidelity_v2_20261015.py','qwen_tail_fidelity_v3_20261015.py',
        'qwen_equal_graph_20261014.py','entity_abc_contracts_20261015.py','qwen_base_entity_abc_data_20261015.py',
        'qwen_base_entity_abc_eval_20261015.py','qwen_base_entity_abc_train_20261015.py',
        'score_strict_units_20261010.py','analyze_qwen_base_entity_abc_20261015.py']
    selected += [(q.ROOT/'code'/name,'code/'+name) for name in names]
    selected += [(p,'logs/'+p.name) for p in (q.ROOT/'logs').glob('qwen_tail_open_*20261015.log')]
    snapshot={name:path.read_bytes() for path,name in selected}
    manifest={name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in snapshot.items()}
    q.dump(OUT/'manifest.json',manifest)
    snapshot['manifest.json']=(OUT/'manifest.json').read_bytes()
    archive=OUT/'qwen_tail_open_suite_20261015.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name,data in snapshot.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=int(time.time())
            tf.addfile(info,io.BytesIO(data))
    result={'archive':str(archive),'sha256':q.sha(archive),'files':len(snapshot),'bytes':archive.stat().st_size,
        'report_sha256':q.sha(target),'new_eval_predictions':621,'sealed311_reads':0}
    q.dump(OUT/'archive_summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


def c_previous():
    import qwen_tail_fidelity_20261015 as c
    return c.PREVIOUS/'eval/base_pair/predictions.jsonl'


if __name__=='__main__':main()

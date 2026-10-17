"""Finish frozen tail-calibration analyses and auditable Chinese report."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tarfile
import time
import qwen_tail_fidelity_20261015 as c
import analyze_tail_fidelity_20261015 as a

q=c.q;RUN=c.RUN
LABEL={'base':'原始Qwen','base_block':'Qwen＋防重探针','anchor':'旧Anchor',
    'base_ce':'新Base-CE','base_pair':'新Base-Pair / Ours'}


def report():
    summary=json.loads((RUN/'final_analysis/summary.json').read_text())
    audit=json.loads((RUN/'compiled/audit.json').read_text())
    names=summary['name_group_audit'];gate=summary['gates'];protocol=json.loads((RUN/'protocol.json').read_text())
    baseline=summary['training_base_diagnostic'];budget=summary['actual_budgets']
    lines=['# EQUAL-RAG：原生句尾校准与完整前缀保护实验', '',
        '日期：2026-10-15（客户端北京时间）；主模型：Qwen3-14B。', '',
        '## 1. 本轮执行结果与目标核对', '',
        f"带局部词面引用支持的名字组：新增{names['locally_cited']['added']}组，丢失{names['locally_cited']['lost']}组，净变化{names['locally_cited']['net']:+d}组。",
        f"用户计数目标“新增≥17且丢失≤3”：**{'达成' if names['requested_count_goal_met'] else '未达成'}**；ΔQ>0：{'是' if names['positive_net_Q'] else '否'}。上轮新增17组中本轮仍命中{names['locally_cited']['old_added_retained']}组，上轮丢失17组中本轮恢复{names['locally_cited']['old_lost_recovered']}组。",
        '这是固定训练门控之后一次性生成的结果，没有以45/24题的Q、丢失组数或长度选择checkpoint。两套数据都是已反复使用的开发集，不是独立盲测；311题继续封存。词面支持、已知别名集合与局部引用匹配均不等于语义关系或事实正确性证书。', '',
        '## 2. 相对提案的必要实现说明', '',
        '- 删除旧Anchor轨迹的15条C和16条B，不把它们假称为原始Base的原生轨迹。',
        '- 固定21条A；它们来自14个原生前缀。每个A正跨度先补完全部已知剩余实体，再设C1收尾，避免剩余实体未补完就强教EOS。C1作为21条A内的终端子标签，不另重复计成21条独立训练回答。',
        '- 已有128条Base轨迹中，没有合格的原生`].<EOS>` C2；实际是裸`]`后EOS。20条C2保留真实原生token路径，不凭空插句点，负样本才是合成的盲目续写。',
        '- 20条C2中10条处于零已知词面支持状态。C2只对原句尾闭合/EOS施加正CE，不把其前面的未知实体视作正确答案来做实体CE。未知gold遗漏仍可能存在，所以这里是原生停止保持，不是语义不可答证明。',
        '- 128条已收集Base轨迹中未发现满足“字面同名重复且后面有新支持实体”的B，因此本轮B=0，成功率N/A；不为凑数量引入Anchor或非重复样本。',
        f"- 所有A/C2均使用与评测相同的FORMAT提示样式。保护120条自然EOS的Base原生轨迹、全部{audit['retention_positions']:,}个保留位点，不再抽样16个位点、不丢弃内部非重复前缀。14个A训练前缀外，另对37条已知剩余集合非空的未标注轨迹开放其原生句尾窗口：不额外教补齐，但不把错误早停EOS锁死为保留目标。内部正常续写不改成C。", '',
        '### 训练目标', '',
        '`监督CE = 首动作CE + 修复跨度平均CE + 首个新实体名字CE + 2 × C1闭合/EOS平均CE`', '',
        '`保留损失 = 10 × KL(Base || Student) + 2 × 翻转位点保序Hinge`', '',
        'KL容忍阈值为0；保序Hinge仅在学生argmax不同于Base时触发，margin=0.5，并只在已翻转位置上求平均，避免被大批正常位置稀释。除此之外每次监督访问另配一次原生保留轨迹访问，辅助保留损失系数0.5。所有保留位置均在完全相同的原生前缀上对齐，不保留teacher forcing后的编辑后缀。',
        'Pair在CE和保留项之外增加结构化Hinge：A的继续/闭合分叉、正确/错误引用数字、C1/C2的闭合/EOS与续写/当前上下文重复名字开头。A起点权重1；引用组权重0.5；终端组权重2，两个终端动作平均后各得到单位权重。没有对跨样本普通实体名词做负向Hinge。CE也保留相同正跨度、首新名字正CE、C1/C2和前缀保护，不人为削弱控制臂。',
        'C1闭合用等价正动作集合`]`或`].`计算类概率，允许Base自然的裸右括号停止，不强迫一个原本已正确停下的上下文先学会加句点。teacher-forcing正轨迹仍保留原来带句点的完整文本，闭合后的EOS由正CE监督；C2仍保持原生token路径。',
        '学习率2e-5；LoRA rank8、alpha16、七类投影、dropout0；梯度累积4、梯度裁剪1。每5步检查门控：A合法首新项成功≥12/21且完整保留池top-1一致率≥97%，首次同时满足即停。两臂独立早停；若步数不同，就不是严格同更新次数的损失消融。最大初始预算60步，不刷满旧800次监督访问。', '',
        '## 3. 训练停止步数、成功率与前缀一致率', '',
        '| 系统 | 更新步数 | 监督访问/辅助保留访问 | A /21 | B | C1 /21 | C2 /20 | C1+C2 /41 | Prefix Top-1 | 保留前缀完全一致 |',
        '|---|---:|---|---:|---|---:|---:|---:|---:|---|']
    for model in ['base','base_ce','base_pair']:
        d=baseline if model=='base' else gate[model]['diagnostic']
        count1=d['C1']['success'];count2=d['by_type']['C']['valid_type_success']
        if model=='base':steps=0;visits='—';fidelity='100%（自身参考）';perfect='120/120（自身参考）'
        else:
            steps=budget[model]['updates'];visits=str(budget[model]['supervised_visits'])+'/'+str(budget[model]['aux_retention_visits'])
            f=gate[model]['fidelity'];fidelity=f"{100*f['prefix_top1_fidelity']:.3f}%";perfect=f"{f['fully_preserved_trajectories']}/120"
        lines.append(f"| {LABEL[model]} | {steps} | {visits} | {d['by_type']['A']['valid_type_success']} | N/A（0条） | {count1} | {count2} | {count1+count2} | {fidelity} | {perfect} |")
    lines+=['', 'A主口径是从原修复起点自由续写后的第一个新实体及引用合法，而非仅命中`],`，也不要求不同正顺序中的某一个指定名字。C1是给定完整补齐前缀后的停止诊断，不代表自由生成一定走得到它。另列完整A链条成功，避免用终点成功代替路径可达：', '',
        '| 系统 | A指定首目标 /21 | A完整已知集合＋闭合EOS /21 |', '|---|---:|---:|']
    for model in ['base','base_ce','base_pair']:
        d=baseline if model=='base' else gate[model]['diagnostic']
        lines.append(f"| {LABEL[model]} | {d['by_type']['A']['conditional_prefix_success']} | {d['by_type']['A']['complete_chain_success']} |")
    lines+=['', '| 系统 | A续写标点 /21 | A独立原生前缀成功 /14 |', '|---|---:|---:|']
    for model in ['base','base_ce','base_pair']:
        d=summary['training_diagnostic_detail'][model]
        lines.append(f"| {LABEL[model]} | {d['A_first_native_continuation_token']} | {d['A_unique_prefix_valid_success']} |")
    lines+=['', '### 每5步的训练门控记录', '', '| 系统 | 步数 | A /21 | C1 /21 | C2 /20 | 前缀top-1一致率 | 达标 |', '|---|---:|---:|---:|---:|---:|---|']
    for model in ['base_ce','base_pair']:
        for path in sorted((RUN/model).glob('gate_*/gate.json'),key=lambda p:int(p.parent.name.split('_')[-1])):
            g=json.loads(path.read_text());d=g['diagnostic']
            lines.append(f"| {LABEL[model]} | {g['step']} | {d['by_type']['A']['valid_type_success']} | {d['C1']['success']} | {d['by_type']['C']['valid_type_success']} | {100*g['fidelity']['prefix_top1_fidelity']:.3f}% | {'是' if g['passed'] else '否'} |")
    lines+=['', '注意：97%是训练保留位点的token加权一致率，不是每条保留前缀完全一致，更不是未见题的100%保证。这里的硬约束是“未达标不进入评测”的门控；KL和Hinge本身仍是软损失，不是推理期强制锁定。', '',
        '## 4. 45题与Raw24的自由生成评测', '',
        '所有系统输入token逐题一致。CE/Pair均为关闭thinking的greedy自由生成，max_new_tokens=256，无阻断、路由、重排或后处理。前三个系统复用相同输入的既有生成缓存。词面Q是可见支持组的micro召回代理，Full F1按完整gold答案计算后逐题平均；tokens含实际生成的终止token。重复率沿用规范化逗号列表口径，可漏计格式损坏及别名重复；真触顶按实际256 token且无EOS的遥测定义。', '']
    for split,title in [('multi45','45题多答案子集'),('raw24','Raw24')]:
        lines+=['### '+title,'','| 系统 | Q % | Full F1 % | 平均Token | 重复率 % | 真触顶率 % |','|---|---:|---:|---:|---:|---:|']
        for model in LABEL:
            r=summary['tables'][model][split]
            lines.append(f"| {LABEL[model]} | {100*r['grounded_quality']:.2f} | {100*r['answer_f1']:.2f} | {r['mean_tokens']:.2f} | {100*r['repetition_rate']:.2f} | {100*r['true_cap_hit_rate']:.2f} |")
        lines.append('')
    lines+=['Raw24只有4题、8个已知可见组贡献Q；其中20题无已知词面支持不等于语义不可答。不能把这个Q当24题全部答案正确率。', '',
        '## 5. 20,000次问题级配对Bootstrap', '',
        '固定seed20261010；两系统按同一问题抽样索引重采样。下面是前者−后者的差值及95%百分位区间；Q、F1、重复、触顶用百分点，Token用绝对数。未做多重比较校正；区间跨0不证明优势，也不证明等效或非劣。', '']
    for split in ['multi45','raw24']:
        lines+=['### '+split,'','| 比较 | ΔQ [95% CI] | ΔFull F1 [95% CI] | ΔToken [95% CI] | Δ重复 [95% CI] | Δ真触顶 [95% CI] |','|---|---|---|---|---|---|']
        for model,base in [(m,'base') for m in LABEL if m!='base']+[('base_pair','base_ce'),('base_pair','base_block')]:
            r=summary['paired_bootstrap'][model+'__minus__'+base+'__'+split];cells=[]
            for field in ['grounded_quality','answer_f1','mean_tokens','repetition_rate','true_cap_hit_rate']:
                x=r[field];scale=1 if field=='mean_tokens' else 100
                cells.append(f"{x['delta']*scale:+.2f} [{x['ci95'][0]*scale:+.2f}, {x['ci95'][1]*scale:+.2f}]")
            lines.append('| '+LABEL[model]+' − '+LABEL[base]+' | '+' | '.join(cells)+' |')
        zero=summary['paired_bootstrap']['base_pair__minus__base__'+split]['zero_Q_denominator_draws']
        lines+=['',f'Q分母为0的抽样有{zero}次，仅从Q区间中排除；其他指标保留全部20,000次抽样。','']
    lines+=['## 6. 重点：名字组新增、丢失及上轮17组的身份追踪', '',
        '| 口径 | Base命中 | 新Pair命中 | 新增 | 丢失 | 净变化 | 上轮新增仍保留 | 上轮丢失已恢复 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind,title in [('name_only','仅名字'),('locally_cited','名字＋局部词面引用支持')]:
        r=names[kind]
        lines.append(f"| {title} | {r['base']} | {r['pair']} | {r['added']} | {r['lost']} | {r['net']:+d} | {r['old_added_retained']} | {r['old_lost_recovered']} |")
    lines+=['', '同一问题内同一名字组只计一次。主Q库存与评分完全一致，45题分母176；未用别的解析规则把失分改成得分。逐题名字、别名、支持文档、旧新增保留与旧丢失恢复详见`final_analysis/name_groups45.jsonl`。', '',
        '## 7. 如何判断路线修正有没有成功', '',
        '需要同时看：旧答案是否保住、新名字是否真正带来支持命中、Raw24是否还膨胀、自由生成是否自然终止，以及Pair是否优于强CE与零训练规则。训练双门控通过只是进入评测的条件，不能替代这些结果。',
        '用户提出的旧C误截断、15.4轮过拟合、缺少句尾反例是合理待检验解释，但不是已经隔离证明的因果根源。旧C来自Anchor的离线轨迹，不能直接称为Base自己原生内部的截断；本轮同时改变数据、完整修复路径、提示样式、保护、引用对比和停止策略，不能从一次联合修正单独归因某一个组件。',
        '训练侧先保留了一次不进入开发评测的试跑：它发现把37条未标注的漏答轨迹旧EOS也放入保留项，会与句尾条件补齐的方向对抗；按全部位置平均保序项又会稀释稀有翻转。修正后从同一官方Base重新初始化，未沿用试跑权重；监督标签逐字节相同，参考缓存仅取原Base分布的精确位置子集。试跑门控、日志与checkpoint保留，试跑开发生成数为0。',
        '第二次训练侧试跑进一步发现，续写标点成功不等于新实体成功，而且C1收尾权重被平均稀释后会退化。因此最终模型从Base重新训练：补上首新实体的纯正CE，平衡A起点与C1终端的动作权重，并把原生裸右括号纳入合法停止类。正实体、引用及完整正token轨迹未变；增加的是CE位置元数据与停止等价类，不是普通实体负样本。第二次试跑同样没有开发集生成。',
        '原生停止保持保住的是Base行为，并未证明Base前缀中的全部实体正确。已知gold集合可能不全；没有词面剩余不等于真实证据穷尽。本轮不引入推理外挂来掩盖这些限制。',
        '如果本轮未达到“新增≥17、丢失≤3”，应直接报告未达标；不能在同一45/24题上继续反复选checkpoint，直到看上去成功。若出现净正收益但区间仍跨0，只能称探索性点估计改善，需要新的独立开发/盲测、多随机种子和分组件消融。', '',
        '## 8. 复现与核验', '',
        '13项单元测试和独立全量标签/保留位点核验通过；模型从官方Base新建LoRA，Anchor仅作历史膨胀对照。封存集读取及生成均为0。',
        f"训练标签SHA256：`{protocol['records_sha256']}`；原生保留数据SHA256：`{protocol['retention_sha256']}`。",
        f"运行目录：`{RUN}`。主代码：`qwen_tail_fidelity_20261015.py`。",
        '复核包保留预测、标签、全部门控、日志、统计和源代码，不含底座、LoRA权重、参考分布缓存张量；完整重训需项目原环境与父轮源数据。历史轮次、失败门控和权重均未覆盖或删除。', '']
    target=RUN/'EQUAL_TAIL_FIDELITY_RESULTS_20261015.md';assert not target.exists()
    target.write_text('\n'.join(lines),encoding='utf-8');return target


def verify_and_bundle(path):
    summary=json.loads((RUN/'final_analysis/summary.json').read_text())
    assert all(g['passed'] for g in summary['gates'].values())
    unit=(q.ROOT/'logs/qwen_tail_fidelity_v3_unit_tests_20261015.log').read_text()
    assert 'Ran 13 tests' in unit and unit.rstrip().endswith('OK')
    independent=json.loads((RUN/'independent_verification.json').read_text());assert independent['status']=='PASS'
    index=json.loads((RUN/'reference/index.json').read_text())
    checks={'status':'PASS','all_gates_passed':True,'unit_tests_passed':13,
        'native_prefix_top1_replay_mismatches':sum(r['argmax_vs_original_greedy_token_mismatches'] for r in index.values()),
        'all_nonrepeat_prefix_positions_protected':sum(len(r['positions']) for r in index.values()),'no_rules_on_main_models':True,
        'same_initial_data_and_script':True,'same_update_budget':summary['same_update_budget'],
        'actual_budgets':summary['actual_budgets'],'sealed311_inputs_opened':0}
    q.dump(RUN/'completed_verification.json',checks)
    q.dump(RUN/'environment.json',{'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','numpy']}})
    shutil=__import__('shutil');shutil.copy2(c.PREVIOUS/'eval/base_pair/predictions.jsonl',RUN/'previous_Pair_predictions69.jsonl')
    q.dump(RUN/'pipeline_status.json',{'state':'COMPLETE','new_dev_generations':138,'no_test_checkpoint_selection':True,
        'count_goal_met':summary['name_group_audit']['requested_count_goal_met'],'sealed311_reads':0})
    selected=[]
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.md'] and not any(s.startswith('checkpoint_') for s in p.relative_to(RUN).parts):
            selected.append((p,'run/'+p.relative_to(RUN).as_posix()))
    scripts=['qwen_tail_fidelity_20261015.py','audit_qwen_tail_native_20261015.py','verify_tail_fidelity_20261015.py',
        'qwen_tail_fidelity_v2_20261015.py','test_tail_fidelity_v2_20261015.py',
        'qwen_tail_fidelity_v3_20261015.py','test_tail_fidelity_v3_20261015.py',
        'test_tail_fidelity_20261015.py','analyze_tail_fidelity_20261015.py','closeout_tail_fidelity_20261015.py',
        'entity_abc_contracts_20261015.py','qwen_base_entity_abc_eval_20261015.py','qwen_base_entity_abc_train_20261015.py',
        'qwen_base_entity_abc_data_20261015.py','qwen_equal_graph_20261014.py','score_strict_units_20261010.py',
        'analyze_qwen_base_entity_abc_20261015.py']
    selected +=[(q.ROOT/'code'/name,'code/'+name) for name in scripts]
    for p in (q.ROOT/'runs/qwen_tail_fidelity_20261015').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl'] and not any(s.startswith('checkpoint_') for s in p.relative_to(q.ROOT/'runs/qwen_tail_fidelity_20261015').parts):
            selected.append((p,'training_only_pilot/'+p.relative_to(q.ROOT/'runs/qwen_tail_fidelity_20261015').as_posix()))
    for p in (q.ROOT/'runs/qwen_tail_fidelity_v2_20261015').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl'] and not any(s.startswith('checkpoint_') for s in p.relative_to(q.ROOT/'runs/qwen_tail_fidelity_v2_20261015').parts):
            selected.append((p,'training_only_pilot_v2/'+p.relative_to(q.ROOT/'runs/qwen_tail_fidelity_v2_20261015').as_posix()))
    selected +=[(p,'logs/'+p.name) for p in (q.ROOT/'logs').glob('qwen_tail_fidelity_*20261015.log')]
    out=q.ROOT/'runs/qwen_tail_fidelity_v3_closeout_20261015';out.mkdir()
    manifest={name:{'sha256':q.sha(p),'bytes':p.stat().st_size} for p,name in selected};q.dump(out/'manifest.json',manifest)
    archive=out/'qwen_tail_fidelity_20261015.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p,name in selected:tf.add(p,arcname=name,recursive=False)
        tf.add(out/'manifest.json',arcname='manifest.json')
    result={'archive':str(archive),'sha256':q.sha(archive),'files':len(selected)+1,'bytes':archive.stat().st_size,
        'report_sha256':q.sha(path),'requested_goal_met':summary['name_group_audit']['requested_count_goal_met']}
    q.dump(out/'archive_summary.json',result);print(json.dumps(result,indent=2),flush=True)


def main(wait=False):
    if wait:
        while True:
            p=RUN/'pipeline_status.json'
            status=json.loads(p.read_text()) if p.exists() else {}
            if status.get('state')=='GENERATIONS_COMPLETE_READY_FOR_ANALYSIS':break
            if status.get('state')=='GATE_FAILED_NO_NEW_DEV_GENERATION':
                print('Training gate failed; no development generation or result fabrication.',flush=True);return
            time.sleep(20)
    a.main();path=report();verify_and_bundle(path)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args();main(args.wait)

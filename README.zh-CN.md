# EQUAL-RAG

针对多答案、带引用 RAG 的训练期轨迹修复研究。

[English](README.md) · [方法](docs/METHOD.md) · [全部实验](docs/EXPERIMENTS.md) · [复现](docs/REPRODUCIBILITY.md) · [完整补充报告](results/supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md) · [中文结论解读](results/supplementary_20261017/SUPPLEMENTARY_INTERPRETATION_20261017.md)

> 更新至 **2026-10-17 补充实验**，包含正负结果，不是已录用论文或确立优越性的声明。
>
> **重要修正：新结果不支持 EQUAL 在 Q 上优于强提示、匹配 SFT/DPO。** 旧 Anchor 的长度膨胀仍有诊断意义，但不足以证明算法优势。

## 我们在做什么

目标不是一味缩短答案，而是保住原有答案与引用，在原生句尾补齐有词面证据支持的新实体，再正常结束。当前框架包括 A/C1/C2 句尾构造、跨 BPE 边界联合重分词、KL + Top-1 保序，以及可选 A-Only Hinge。原始 Base 是初始化和保留参考，旧 Anchor 只是历史对照。

主方法推理始终为**未合并 LoRA + 标准贪心自由生成**，关闭 thinking，最多 256 个新 token；没有路由、外挂规则或真实答案。只有单列的 Entity Blocker 基线使用防重规则。词面支持不等于语义蕴涵或证据已穷尽。

## 最新结果：固定主 Seed、Full311

Q/F1 为百分数，长度为生成 token 数。

| Qwen3-14B 系统 | Q | Full F1 | Token | 新增/丢失/净增支持引用组 |
|---|---:|---:|---:|---:|
| Base | 51.86 | 12.40 | 26.23 | 0/0/0 |
| 穷举提示，无训练 | **70.45** | 12.68 | 40.67 | 107/17/+90 |
| 匹配 60 步整答案 SFT | 65.91 | 13.76 | 35.49 | 78/10/+68 |
| 匹配 60 步序列 DPO + 0.1 SFT | 65.70 | 13.75 | 38.21 | 81/14/+67 |
| EQUAL CE β=0 | 57.44 | 13.34 | 29.54 | 37/10/+27 |
| EQUAL AOnly β=0.25 | 60.33 | **13.78** | 31.17 | 51/10/+41 |
| EQUAL AOnly β=0.50 | 60.74 | 13.60 | 30.53 | 54/11/+43 |

β=0.25 相对穷举提示、SFT、DPO 的 ΔQ 为 −10.12、−5.58、−5.37 个百分点，名义 95% 配对区间均低于 0；其 Full F1 差值区间均包含 0。更短不等于综合性能显著更优，区间跨 0 也不是等效性检验。

Mistral-7B 上 AOnly 将 Q 从 **37.81% 提升至 45.25%**，ΔQ 区间 **[+2.28,+13.49]** 个百分点，Full F1 从 8.99% 到 10.89%。跨家族相对 Base 的收益得到支持，但匹配 SFT 达到 **52.48% Q / 12.03% F1**；不能声称胜过 SFT。AOnly 更少丢失旧组、更少重复/触顶，新增也较少，是权衡而非全面胜出。

七个命名 tokenizer/encoding 在原 21 条 A 记录上均存在独立拼接与联合分词不一致。Qwen2.5/3 不是独立家族，本地 Mistral 副本不重复计数。这支持检查局部修复的分词边界，**不证明普遍硬死锁或“只有训练才能续写”**。

完整表格、五个切片、20,000 次配对 Bootstrap、Gnew/Gfix/L/Net、训练命中和 Ret./120 见[完整报告](results/supplementary_20261017/EQUAL_SUPPLEMENTARY_RESULTS_20261017.md)。匿名逐题数值可独立重算统计，不包含回答或检索文本。

## 不要混淆两阶段结果

- 原三 Seed 主盲测：12 系统、3,732 条生成，3 配置×3 Seed，内部预注册、零新训练、零挑 checkpoint。β=0.25 的三 Seed Full311 均值为 Q **58.13 ± 2.01%**、F1 **13.48 ± 0.26%**。
- 上表补充实验：固定主 Seed `20261015`；新基线和 Mistral 只有单 Seed，311 已解封，不能再次称独立盲测。
- 两阶段数字不同不是改写旧结果：统计目标不同，原值保留。[主盲测](results/reports_20261015/EQUAL_HELDOUT311_FINAL_RESULTS_20261015.md) · [三 Seed 与 12 项消融](results/reports_20261015/EQUAL_MULTI_SEED_ABLATION_RESULTS_20261015.md)。
- 数据为 ALCE/QAMPARI 内部划分，不是官方榜单；没有人类语义标注或预训练无污染认证。
- 同为 60 次更新、240 次监督访问，但 EQUAL 多 240 次辅助保留访问；不等于 FLOPs 或壁钟时间完全匹配。

## 无 GPU 查看与核验

```bash
python scripts/show_results.py
python scripts/verify_release.py
python -m pip install -r requirements-analysis.txt
python scripts/run_checks.py
python scripts/reproduce_supplementary.py
```

最后一个命令可重建 325 个点值、65 份得失汇总、1,425 个差值/区间数值。它验证发布统计，不替代原始文本评分、模型推理或语义核查。

GPU 复现需要另行取得底座、原数据、冻结标签和参考缓存；不是无需准备即可启动的训练 SDK。[复现说明](docs/REPRODUCIBILITY.md) · [数据政策](docs/DATA.md) · [结果索引](results/README.md)。

不上传密码、私有服务器信息、原题/检索段落/完整生成、真实回答 token 序列、模型权重或日志压缩包。原归档 SHA、公开数值和发布转换 SHA 均保留。旧实验不删除，按日期标注。

仓库可见不代表开源许可；目前未授予开源许可证，也不声明论文已被录用。[许可状态](LICENSE_STATUS.md) · [引用](CITATION.cff)。

# EQUAL-RAG

**研究目标：让模型依据检索证据，回答完整、引用正确、不重复，并在完成后自然停止。**

[English](README.md) · [方法详解](docs/METHOD.md) · [实验结果](docs/EXPERIMENTS.md) · [复现说明](docs/REPRODUCIBILITY.md) · [变化历程](docs/HISTORY.md)

> 截至2026-10-11的研究开发快照，当前优先模型为Qwen3-14B。本轮实验已完成，但“主方法稳定优于强对照”的目标尚未达成。本仓库不是已录用论文的官方发布，不保证效果或投稿结果。

## 先用一个例子理解

假设问题需要三个答案，证据支持 `Alice [1], Bob [3], Carol [2]`。

- 只输出 `Alice [1]` 很短，但漏了必要答案。
- 输出 `Alice [1], Bob [2], Carol [2]`，长度合适，但Bob的引用不对。
- 不断重复三个答案，完整却浪费生成成本。

我们希望训练模型识别这些差异：**已有的有效答案要保留，遗漏的要补齐，引用要对应，完整后才结束。**训练时用答案—证据标签构造修复目标，并用选择性KL约束保护原有内容；推理时正常生成，不做方法路由，不把标准答案交给模型。“EQUAL”尚无正式英文全称，不自行编造缩写展开。

## 已完成的工作

1. 早期输出长度节省、长度匹配和跨模型试验；与当前RAG实验分开保存。
2. 答案—证据反事实训练、模型自身生成前缀挖掘、补齐/停止标签、选择性KL、LoRA训练与真实token结束记录。
3. 最新Qwen三臂：主方法、同预算CE、同预算UL；350例训练，单seed，共480条开发预测。
4. 七系统1120条已有回答自动引用筛查，943对去重NLI输入，截断0。
5. 训练位点诊断、未插值模型诊断、逐题配对bootstrap、多重比较敏感性分析和小样本延迟回放。
6. 实验来源与哈希核验、标准化文档和可在无GPU环境运行的查看/检查工具。

本发布只包含研究代码、合成样例与聚合结果，不上传原始检索文本、逐题预测、模型权重、teacher缓存、实验日志和未投稿论文。

## 当前结果：有局部改善，尚未得到稳定验证

| 两组开发数据各80题 | 方法 | 词面引用覆盖Q | 平均token | 完整答案F1 |
|---|---|---:|---:|---:|
| 原开发集 | 已训练anchor | 47.285% | 42.513 | 33.664% |
| 原开发集 | 当前pair | 47.964% | 39.913 | 33.802% |
| 原始top5检索开发集 | 已训练anchor | 40.580% | 122.538 | 6.083% |
| 原始top5检索开发集 | 当前pair | 47.826% | 131.375 | 6.397% |

Q是**词面、局部引用覆盖代理**，不等于语义事实正确率。两组问题不同，不能拿差异当同题检索条件的因果消融；两个集合都反复用于开发，也不是官方排行榜测试。

综合效率U=Q/平均token，pair相对anchor：原开发+8.04%，95%区间[−0.66%,+25.46%]；原始检索+9.93%，区间[−10.30%,+30.02%]。均跨0，也尚未稳定优于匹配CE/UL。

实现上修复了72处监督冲突，但72个真实停止前缀仍没有一次把EOS选为最高概率动作。自动NLI未确认稳定语义改善，小样本延迟未证明加速。当前版本封存集预测0、人工标注0；旧错误EOS版本的多个seed不能当新版本确认。

这让下一步更明确：研究从真实分歧处连续学习“闭合—补齐答案及引用—EOS”，而不只学习两个孤立位点。**新方案尚未启动。**完整三臂结果及判断见[实验台账](docs/EXPERIMENTS.md)。

## 无GPU快速查看

仓库为私有，需要先获得访问权限。以下命令只查看已保存结果，不训练模型：

```bash
git clone https://github.com/Jjil11/EQUAL-RAG.git
cd EQUAL-RAG
python scripts/show_results.py
python scripts/verify_release.py
```

安装轻量分析依赖后运行合成样例单元测试：

```bash
python -m pip install -r requirements-analysis.txt
python scripts/run_checks.py
```

建议先创建虚拟环境。训练需Linux/CUDA及外部数据、prepared labels、anchor adapter、缓存和模型；这些并未打包，不能宣称一键精确复现所有原实验。详见[复现指南](docs/REPRODUCIBILITY.md)。

发布前已通过19项本地CPU单元测试，以及153项代码/结果清单哈希和149个Python文件语法检查。测试环境与边界见[VALIDATION.md](docs/VALIDATION.md)；不冒充远程CI通过或全量GPU训练复现。

## 推荐阅读顺序

- 不熟悉方向：[METHOD.md](docs/METHOD.md)，从token、证据边、补齐和停止讲起。
- 想知道结果是否有效：[EXPERIMENTS.md](docs/EXPERIMENTS.md)，看所有对照和置信区间。
- 想知道方法如何变化：[HISTORY.md](docs/HISTORY.md)，区分有效、历史和无效版本。
- 想运行代码：[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)与[DATA.md](docs/DATA.md)。
- 想继续工作：[ROADMAP.md](docs/ROADMAP.md)，先稳定Qwen收益，再最终确认和扩模型。

## 目录与权限

`research/`是保留日期的实验实现；`scripts/`提供轻量入口；`results/`保留聚合指标；`historical/token_saving/`保存早期非RAG代码；`release_manifest.json`记录原始/导出代码哈希。发布版的机器路径改为环境变量，算法不因此被重新验证。

不上传密码、token、私有服务器地址、SSH工具、权重、原始数据或论文草稿。数据与模型须遵守各自许可，参见[DATA.md](docs/DATA.md)。目前**暂未授予开源许可**，可见或可访问不等于允许任意复用，参见[LICENSE_STATUS.md](LICENSE_STATUS.md)。引用本软件可使用[CITATION.cff](CITATION.cff)，不虚构论文、DOI或录用信息。

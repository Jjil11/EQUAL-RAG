# Method guide / 方法详解

## 1. The problem in plain language

A retrieval-augmented model receives a question and retrieved documents, then emits answers with citations. It generates one token at a time. Saving tokens is useful only if the model preserves the answers the question needs and cites the documents that support them.

本项目不是单纯“让回答越短越好”。我们希望模型先知道什么内容必须保留，缺什么就补什么，完成后再停止；“少说”必须与“答全、有依据”一起检验。

## 2. Answer–evidence edges

A training edge links an answer group (including aliases) to a supporting retrieved document. For the synthetic response `Alpha [1], Beta [2]`, the two required edges are Alpha→1 and Beta→2. Omitting Beta, citing Alpha to the wrong document, or repeatedly emitting Alpha are different failures.

In this implementation, support labels use answer aliases and upstream document annotations. The presence of an alias in a paragraph does not establish the relation asked by the question. This is a practical training/evaluation proxy, not a proof of semantic truth. Gold labels belong to training compilation and evaluation; the inference policy never receives gold answers.

## 3. The implemented method family

| Component | Role | Implementation pointers |
|---|---|---|
| Evidence labels / counterfactuals | Contrast complete answers with omission, wrong citation, or repetition | `build_qampari_pilot.py`, `completion_pairs_20261010.py` |
| On-policy prefixes | Study the states actually visited by the current model | `mine_qampari_trajectories.py`, `mine_qwen_retrieval_align_20261010.py` |
| Completion targets | Preserve existing covered items, add missing supported items, then terminate | `completion_pairs_20261010.py` |
| Selective anchor protection | Avoid changing useful behavior outside the repair region | `train_qwen_priority_20261010.py`, `train_qwen_terminal_bridge_20261010.py` |
| Native-token consistency | Align first action and terminal EOS; avoid conflicting positive/negative paths | `compile_qwen_native_consistent_20261010.py` |
| Telemetry | Record exact generated IDs, actual EOS, and safety-cap termination | `eval_verified_outputs_20261010.py` |

All pointers above are in [`research/`](../research). They describe a sequence of experimental versions, not a single combined model proven by every historical result.

## 4. Latest training objective

For an active repair, the current pair arm uses:

```text
loss = 10 × max(selective_KL − 0.0001, 0)
     + 0.1 × (first-action allowed-set CE + margin hinge)
     + 0.1 × terminal allowed-EOS-set CE
```

The hinge compares the best allowed next-token log probability with the recorded rejected next-token log probability, with margin 1. CE uses total probability mass over allowed tokens. The CE control removes the first-action hinge. The UL control uses allowed-set CE plus negative-action unlikelihood instead of that hinge. All three retain the same terminal term, labels, anchor cache, optimizer budget and selected training rows.

这是在指定训练前缀上鼓励正确下一动作，同时用KL保护保留部分；并不是对所有事实正确性或完整性的硬保证。最新实现只监督首/末稀疏位点，并未完整监督中间所有补齐文本。缓存的KL坐标采用重分词后的响应位置，不应与原始生成ID坐标混为一谈。

## 5. What changed in the latest repair?

The invalid preceding native-terminal version rejected a punctuation token on a first-action path while retaining that same token on a positive terminal path. The latest compiler jointly aligned 72 native stop contexts and first actions, reducing exact positive-path conflicts to zero. The other 78 counterfactual terminal contexts stayed unchanged. Four decoded-prefix flags under alternate tokenization remain; zero exact conflicts is not an all-label semantic validation.

This is a validity repair, not an independent novelty contribution. EOS and boundary fixes cannot by themselves support a claim of a new research method. The coverage-protection/completion idea must win against credible controls and generalize.

## 6. Why the current result is not the endpoint

At the 72 original stop prefixes, every compared arm still chose a non-EOS token by argmax. The un-interpolated pair checkpoint also produced 0/72 EOS argmax. Many prefixes end after merged `],`, a syntactic state that invites more list items. Teacher-forced completed targets may be easy to terminate but not states reached by free generation.

The next **proposed, unlaunched** experiment is coherent repair-span supervision: begin at the real divergence/boundary, learn a valid closure or missing answer/citation sequence, then EOS. This retains ordinary inference and coverage protection. It is a hypothesis to test, not an achieved result or guaranteed solution.

## 7. Vocabulary / 术语

- **LoRA / adapter**：只训练少量附加参数，不完整重训14B基座。
- **Anchor**：保护和比较用的已训练参考模型；不是原始未适配Qwen。
- **KL**：约束新模型的预测分布别离参考模型太远；不是语义保证。
- **EOS**：模型决定结束的特殊token。
- **cap256**：最多允许生成256个token的安全上限，不是固定回答长度。
- **Development**：可用于方案选择的数据，反复用过后不能再当独立确认。
- **Bootstrap CI**：按问题重采样的区间；单seed的区间不能估计训练seed变化。

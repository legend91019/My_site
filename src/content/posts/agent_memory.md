---
title: "对 agent memory 方向的文献调研和思考"
summary: "主要是从 Stanford CS329A 的方向出发，并进行一轮完整的思考和找 idea"
date: 2026-09-01
tags: ["agent", "memory"]
category: "论文解读"
draft: false
---

【感觉还是得输出一些东西啊，光读文献的话感觉读不进去，理解也不深，还是得强迫自己写点东西，本篇文章主要是泛读，旨在快速入门领域】

## 一、[Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)


**这篇文章的核心简单来说就是：大家一直以来测试模型能力，包括 Benchmark，都是让模型回答一次。但是如果让模型多次尝试，看在 $k$ 次尝试下的 acc【这也在文章中被称为 Coverage，覆盖率】，能提高非常多，并且似乎可以建模。**

那有些观众可能就会说了，你 $k$ 次尝试模型参数是不变的，大模型本质是一个复杂的固定系数算概率，按理来说哪一次输出都不会有差别的。

当然这是 `temperature=0` 的情况。

`temperature=0` 就是选择词表中对应概率最大的分词嘛，不过本文选择 `temperature > 0`，这样模型会按照经过 temperature 调整后的概率分布进行随机采样，每次的输出自然就会产生差异了。并不是只在“概率最大的 temperature+1 个分词”里选择。

**本文其实也是对 scaling law 的一次探索。**

我对 scaling law 比较多的了解是从 DeepSeek LM 来的，简单来说就是模型参数量、数据量、计算量、数据质量、超参数的关系。比如大家所熟知的参数量越大模型越牛，参数量提升要伴随数据量的提升（不然模型表现反而下降），固定计算量下怎么分配参数量和数据量，计算量和最优 `batch_size`、`lr` 的关系，计算量和数据量、参数量的关系（比较旧的估计是 $C=6ND$，$C$ 是计算量，$N$ 是参数量，$D$ 是数据量。我记得 CS336 也提过这个相关的计算，用纯参数量估计的，当然 DeepSeek LM 认为这个估计失真，重新引入了指标 $M$（Non-embedding FLOPs/token））。

说回这篇文章，为什么说同样是对 scaling law 的探索呢？因为这篇文章发现了一个比较好的结果，就是在 abstract 中提到的：

> SWE-bench Lite：开源模型 DeepSeek-Coder-V2-Instruct 单次尝试成功率仅 15.9%。但在采样 250 次后，覆盖率直接飙升至 56%，大幅超越了当时混合 GPT-4o + Claude 3.5 Sonnet 的单次尝试 SOTA 纪录（43%）。

这就带来一个问题：

**在特定任务种类下，小模型多次尝试如果用更少的计算量能胜过大模型，是不是用小模型更加“好快多省”？**【其实这不也是我们现在用 agent coder 的一个权衡吗，你是用免费的豆包多 debug 几次 + 人工 review，还是用 GPT-5.6 Sol 直接解决问题。这个时候 debug 同样和这里的“特定任务种类”有异曲同工之妙，有的 bug 差模型越 debug 越不行，还会给你惹很多麻烦；有的 bug 用差模型解决却更省。】
换句话说就是，**把更多计算量从 training 阶段放到 inference 阶段。**

本文两大指标：
**覆盖率（Coverage，即 $pass@k$）**：定义：当对同一个问题独立采样 $k$ 次时，至少有一个生成样本是正确的问题比例。核心疑问：随着采样次数 $k$ 的增加，覆盖率能否持续增长？

**精确度（Precision / Verification）**：定义：能否从生成的众多候选答案中，准确挑选出那个正确的答案？核心疑问：在没有绝对正确的验证器时，现有的挑选方法（如多数投票、奖励模型）能否跟上采样扩展的步伐？

测试数据集有两种。

**有自动 verifier 的任务：**

- SWE-bench Lite（软件工程）：真实 GitHub issue 修复，用测试用例验证代码。
- CodeContests（竞赛编程）：来自 Codeforces 等平台的算法题，用隐藏测试用例验证。
- MiniF2F-MATH（形式化数学）：用 Lean 4 语言编写的数学证明，由 Lean 编译器自动检查证明逻辑。

这里的 SWE-bench Lite 有一个容易忽略的细节：论文里的“一次尝试”不是一次简单的文本回答，而是一整段模型和 Moatless Tools 之间的多轮交互轨迹。

**没有自动验证器的任务：**

- MATH & GSM8K（数学应用题）：自然语言解答，缺乏编译器，需依靠人工、投票或奖励模型（RM）来判定【我反正是不信一个样本 1000 甚至 10000 个回答版本让人工去评审，但是本文似乎说投票和奖励效果不好？】。论文在计算 Coverage 时使用的是 oracle verifier，也就是只要样本里有正确答案就算覆盖；这更接近上限，不是实际部署时可以直接获得的成绩。

**实验：Coverage Scaling，看 $k$ 增加时 Coverage 的比例。**

实验结果：

1. Coverage 随着 $k$ 增加呈现出极强的对数增长态势，并且没有提前饱和的迹象。
2. 小模型的成长潜力：Pythia-160M 在 MATH 数据集上，$pass@1$ 仅为 0.27%，但在采样 $k=10,000$ 次后，Coverage 达到了 57.0%。Gemma-2B 在 CodeContests 上，$pass@1$ 为 0.02%，当采样到 $k=10,000$ 时，Coverage 达到了 7.1%（性能翻了 350 倍）。不过这不是所有模型都成立：Pythia 系列在 CodeContests 上即使采样到 10,000 次仍然是 0 coverage，作者推测这与代码相关训练数据较少有关。
3. 小模型 + 多次尝试打赢大模型单次：在 SWE-bench Lite 上，开源的 DeepSeek-Coder-V2-Instruct 单次尝试（$pass@1$）为 15.9%。但当 $k=250$ 时，DeepSeek 的 Coverage 直接飙升至 56.0%，一举超越了由 GPT-4o + Claude 3.5 Sonnet 组合维持的单次尝试 SOTA 纪录（43.0%）。
4. 对“采样 $k$ 次能拿到多少 Coverage”做了数学公式拟合：

$$
c \approx \exp(a \cdot k^b)
$$

其中 $c$ 为 Coverage，$k$ 为采样次数，$a<0$、$b<0$ 为拟合参数。
5. 等算力预算下的模型选择：MATH / GSM8K（数学推理）：小模型 + 大采样量更占优。例如，在相同算力预算下，给 Gemma-2B 采样数千次获得的 Coverage，远高于给 Gemma-7B 采样数十次。CodeContests（长文本 / 复杂逻辑代码）：大模型 + 少采样量更占优。因为小模型由于上下文理解能力限制，极难生成高难度的算法框架，此时增大模型规模收益更高。API 金钱成本实测（SWE-bench Lite）：对 DeepSeek-Coder 采样 5 次的解决率（29.62%）高于单次 Claude 3.5 Sonnet（26.70%），而实际的花费仅为 Claude 的 1/3。这里的成本比较是在相同的 Moatless Tools agent 框架下进行的，FLOPs 也只是近似成本，不能直接推广成“小模型永远更便宜”。

论文还专门比较了 Coverage 和实际的答案选择方法：在 MATH 上，Llama-3-8B-Instruct 的 oracle Coverage 从 100 次采样时的 82.9% 增加到 10,000 次时的 98.44%，但多数投票或 reward model 选出的最终答案只从 40.50% 增加到 41.41%。这正是上面 Coverage 和 Precision 之间的差距。


## 二、[An Architecture Search Framework for Inference-Time Techniques](https://arxiv.org/abs/2409.15254)


**背景与动机：**

Inference-Time Techniques 兴起，指的就是在 inference 阶段投入更多计算量。比如前面第一篇论文说的重复采样、多模型集成、多轮修正、自我批评、验证等【有点 harness 的意思了】，“推理算力换性能”是可行的。

**痛点和挑战：**

1. 不同技术在不同任务表现差异大【说到这里我想起来 MCE 这篇论文了，用 meta agent 修改 skill 来达到不同任务使用不同的上下文策略，感觉解决的问题和这个很接近】。
2. 技术间的相互作用，各组件协作缺乏系统理解。
3. 推理组件非常丰富，还有很多选择，比如采样次数、融合策略，搜索空间极大，难以人工手动调优。

**核心贡献：** 提出 ARCHON，一个模块化且自动化的推理搜索框架。

### 1. ARCHON 的模块化 LLM 组件


ARCHON 将推理阶段的技术解耦为多个功能明确的 LLM 组件（Layer）：

**Generator（生成器）**：输入提示词，并行生成候选回答（支持单模型多次采样或多模型集成生成）。

**Fuser（融合器）**：将多个候选回答进行整合，生成更高质量的综合输出。

**Ranker（排序器）**：对候选回答进行两两对比或打分排序，筛选出 Top-K。

**Critic（批评器）**：分析候选回答的优缺点，为后续的排序或融合提供参考意见。

**Verifier（验证器）**：分阶段推导和验证候选回答的逻辑正确性。

**Unit Test Generator & Evaluator（单元测试生成与评估器）**：生成测试用例并运行评估，多用于代码及逻辑推理任务。
【现在回过头来想想这个可以说是 harness 的雏形了：生成器、上下文合并策略、验证器、单元测试。】

### 2. 自动化架构搜索（Architecture Search）


ARCHON 借鉴了神经架构搜索（NAS）的思想：

- 输入：目标基准数据集、推理调用预算（Inference Call Budget）、可用 LLM 列表、候选推理技术。

- 搜索算法：采用贝叶斯优化（Bayesian Optimization）在多达数千种可能配置的设计空间中高效探索（涵盖生成器数量 / 采样数、融合层数、是否加入批评 / 验证器等维度）。

- 约束控制：支持限制 Token 消耗量或 API 调用次数，直接搜索出在给定算力预算下的最优架构。

论文在多个 instruction-following、reasoning 和 coding benchmark 上测试，最好的 ARCHON 架构平均比基线高 15.1%，同时减少约 20.0% 的 inference calls、15.1% 的输入 token 和 13.5% 的输出 token。这个结果依赖具体的模型集合、任务和预算，不是任意组合都能得到同样的提升。
【这个下面详说，其实刚看到感觉是个“请输入文本”的东西啊，毕竟现在 harness 的这些策略如果我的认知没问题的话都是固定的吧，我感觉他们没法实现 claim 的东西。又想了一下，他们经过模块化的话应该是有通用协议的，做 benchmark 似乎也不难。但是仔细想想感觉其实有点鸡肋：一是必须有统一的框架，没有办法再多出组件，他们本身也没有打破原有的挑战——“推理组件非常丰富，还有很多选择，比如采样次数、融合策略，搜索空间极大，难以人工手动调优”。以及不可能在真实工作中先网格搜索一下本任务的最佳组件方案吧，又没有 val 集。】

这里补充一个论文里的限定：ARCHON 的搜索空间虽然由预先定义的组件和连接规则构成，但框架本身声称可以通过统一接口扩展到新的模型、任务和 inference-time technique。也就是说，它不是只能搜索现在这几个组件；只是新组件必须先被写成符合框架协议的模块。

核心实验结论：【都是很符合直觉的东西感觉】

1. 增加多模型集成（Ensemble）和重复采样能显著提升性能。
2. 多层级的批评（Critic）+ 融合（Fuser）组合，相比单纯选择单次最佳回答，能突破性能上限。
3. 指令遵循与文本生成类任务：极度依赖排序（Ranker）和多层融合（Fuser），多次修正能大幅提升回答丰富度与风格。
4. 代码与严密推理类任务：采样数量（Repeated Sampling）以及单元测试（Unit Test）/验证器（Verifier）更为关键。例如在 CodeContests 的一个实验设置中，加入 Unit Test Generator / Evaluator 后，Pass@1 从 17.9% 提升到 29.3%，相对提升约 56%。【我一看这不是上面论文的结论吗，他们应该是直接拿来用了】


## 三、[Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)


**核心观点：**

Inference 阶段分配额外的计算量（或者说计算资源），在一些任务上比单纯扩大参数量或者预训练计算量更加高效。根据输入提示词的难度动态调整测试时计算资源，较小的模型能够达到或超越参数大 14X 的模型的准确率，并能减少 4X 计算量。这里的 14X 和 4X 都是论文在特定模型、MATH 任务和 FLOPs 对齐实验中的结果，不是普遍规律。

测试时计算被归结为两个互补方向：

1. **修改候选生成分布（顺序修正 / Sequential Revisions）**：不再独立并行采样多个回答，而是使模型可以结合上下文中的历史错误解答，迭代修复先前的尝试。论文这里使用了专门训练的 revision model，不只是给原模型重复发送“请再想一遍”【迭代的想法很好，但是往往要面对注意力不够了这个问题。独立取样当然也有第一篇论文所说的 Precision 问题，从众多候选答案中选择正确的 / 最好的。有点像 CNN 改进为 RNN 了有感觉吗】。
2. **优化验证器（基于过程的搜索 / Process-Based Search）**：利用蒙特卡洛采样训练过程奖励模型（PRM），对中间推理步骤逐一评估。在此基础上配合高级树搜索算法（如束搜索 Beam Search、前瞻搜索 Lookahead Search）进行策略寻优。

**关键实验发现与结构化洞察：**

1. 题目难度是分配计算量的关键指标。不同测试时计算策略的有效性极度依赖于问题相对于基座模型的固有难度：简单问题最适合采用顺序修正（局部微调）；困难问题更适合采用并行探索或基于 PRM 的树搜索（如 Beam Search），以探索更多样化的高层解题思路。论文的实验支持这种策略差异，但并不是简单地断言“简单问题上的树搜索一定会过拟合”。
2. 计算最优 scaling 策略（Compute-Optimal Strategy）根据预测的问题难度，动态选择最优的测试时计算组合（平衡并行采样数、顺序修正次数与束搜索宽度），可极大提升推理效率：

| 方法类别 | 对照基线 | 计算最优策略带来的节省 |
| --- | --- | --- |
| 基于 PRM 的搜索 | 并行 Best-of-$N$ | 仅需约 $4\times$ 更少的计算量即可达到或超越 Best-of-$N$ |
| 候选修正 | 并行采样 | 仅需约 $4\times$ 更少的计算量即可超越并行 Best-of-$N$ |
3. 预训练计算量与测试时计算量的等效性（FLOPs 对齐评估）。在总 FLOPs 相同的条件下，将“小模型 + 最优测试时计算策略”与“大 $14\times$ 的模型 + 标准 Greedy 解码”进行对比：

   - 简单与中等难度问题：将额外的 FLOPs 分配给测试时计算，效果优于直接预训练一个大得多的模型。
   - 极高难度问题：如果问题完全超出了基座模型的知识 / 能力边界（Pass@1 接近于零），预训练计算量依然占据主导地位。因此，测试时计算与预训练计算并非在所有难度层面上都能 1:1 完全替代。

论文还提醒了一个实际成本：预测题目难度本身也需要计算。作者用 PRM 的预测近似 difficulty，并用 two-fold cross-validation 避免直接在同一批测试题上选择和评估策略；真实部署时，还要权衡“先花多少计算判断难度”和“直接开始解题”。


## 四、[How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578)


稍微说一下吧，第一篇论文的重复采样的思路似乎和“无限猴子定理”有点关系。这个定理简单来说就是无数个猴子每个猴子都在打字机上随机乱摁，无穷岁月过去总有一只猴子能打出一本莎士比亚。其实就是说重复数量无限，极小的概率也会变成必然，可能这让作者想到了 inference $k$ 次能让像猴子一样的小模型发挥出非常大的潜力吧。为什么在这里说一嘴呢，是因为注意到这篇文章也带 Monkeys。


### 概念背景与核心谜题（The Puzzle）


现象：近期研究（如 Large Language Monkeys 和 Best-of-N Jailbreaking）发现，给大模型多个独立尝试的机会（尝试 $k$ 次，只要有一次成功即算成功），模型的平均成功率的负对数（$-\log(\text{pass}_{\mathcal{D}}@k)$）与尝试次数 $k$ 在对数-对数坐标下呈幂律（Power Law）衰减关系：

$$
-\log(\text{pass}_{\mathcal{D}}@k) \propto k^{-b}
$$

悖论 / 谜题：从概率论直觉推导，对于单道具体问题，假设其单次成功概率为 $\text{pass}_i@1$。因为每次尝试独立，全错的概率是 $(1 - \text{pass}_i@1)^k$。当 $k$ 较大时，根据泰勒展开，单题的负对数成功率随 $k$ 应该呈指数级（Exponential）下降：

$$
-\log(\text{pass}_i@k) \approx (1 - \text{pass}_i@1)^k
$$

核心疑问：既然“单题目”的失败率是指数衰减的，为什么“多题目聚合（Dataset Aggregate）”后的平均失败率会变成幂律衰减？

作者通过数学证明和实证研究解答了这个谜题：宏观的幂律分布来源于“单题指数衰减”与“问题难度分布的长尾/重尾（Heavy-tailed）特征”的结合。

**A. 单题近似呈指数衰减。**作者在 MATH 数学数据集和 HarmBench 越狱测试集上进行了验证，发现对于单个问题 / Prompt，负对数成功率随尝试次数近似指数衰减。

**B. 充要条件：单次成功率的尾部分布决定了幂律。**论文提出了两个核心定理（Theorem 3.1 & 3.2）：在一定的正则条件下，整个数据集的平均性能呈现指数为 $b$ 的幂律衰减（即 $\propto k^{-b}$），当且仅当数据集内各题目单次成功概率 $\text{pass}_i@1$ 在接近 $0$ 处的概率密度函数 $p(p)$ 具有幂律左尾特征（即 $p(p) \propto p^{b-1}$）。直观理解：测试集中总是存在极少数“极难”（$\text{pass}_i@1$ 极其接近 0）的题目。虽然每道题都在被指数级快速解决，但由于这些极难题目的存在“堆积”在尾部，拖慢了整体的解决进度，最终将宏观的整体趋势“扭曲”成了幂律曲线。

**C. 解释异常现象。**之前的研究发现 Meta 的 Llama 3 8B IT 模型在越狱测试中没有呈现幂律分布，而是下降得极快。论文解释：在实验预算范围内，Llama 3 8B 几乎对所有 Prompt 都能被成功越狱，其单次成功率分布不存在接近 0 的长尾。没有这个极难题目的长尾分布，宏观的幂律现象自然就消失了。

**实际应用：全新的“基于分布的 Scaling 预测器”。**传统预测推理算力 Scale 效果的方法是：在不同的 $k$ 值下采样，画出 log-log 图，用最小二乘法（Least Squares）拟合直线。这种方法极其消耗推理算力。

作者提出了一种全新的基于分布的估计器（Distributional Estimator）：只需在小样本 / 少次数下测量模型的单次成功率 $\text{pass}_i@1$ 的分布，拟合一个 Beta 分布或 Kumaraswamy 分布。预测时，直接通过积分数学推演求出任意大 $k$ 值下的性能。相比传统拟合方法，达到相同的预测精度可以节省 2~4 个数量级的推理算力（Inference Compute），相对误差降低一个数量级。

这里有一个实际难点：如果 $\text{pass}_i@1 < 1/N$，用 $N$ 次样本几乎无法分辨它的精确值。作者把这部分看成左尾区间，用区间内的问题比例拟合带尺度参数的分布，并对离散后的分布做最大似然估计。论文的“相对误差降低一个数量级”和“节省 2~4 个数量级算力”主要来自合成数据回测与估计器比较，不能理解为所有任务都能直接获得同样收益。


## [Shrinking the Generation-Verification Gap with Weak Verifiers](https://arxiv.org/abs/2506.18203)


Weaver（弱验证器聚合框架）是一种弱监督算法框架，核心目的是解决 LLM 在 inference 和生成过程中的“生成-验证差距”（Generation-Verification Gap）。

在使用 LLM 进行多次采样生成候选答案时（如针对同一个问题采样 100 个回答），正确的答案往往已经包含在候选集中。但现有的单一“弱验证器”（如 LLM Judge 或奖励模型 Reward Model）往往存在偏差、噪音大和校准度差的问题，难以精准挑出那个正确答案。【第一篇论文中提到的 Precision 问题】

Weaver 可以在不使用 Ground Truth（真实标签）对聚合器进行监督微调的前提下，将多个不完美的弱验证器进行组合，大幅提升筛选准确率。

### 核心工作机制


- 弱监督学习聚合（Weak Supervision Aggregation）：Weaver 将不同的奖励模型和 LLM Judge 视为图模型中的“投票者”。利用弱监督统计学中的矩匹配（Moment Matching），在没有正确答案标签的情况下，反向推算出各个验证器自身的潜在准确率。
- 分数标准化与过滤：验证器的输出形式各异（Logits 概率、0/1 判定、1-5 分打分）。Weaver 将这些分数进行归一化与二值化，并过滤掉输出噪声大、信息量低的无效验证器。
- 加权组合与排序：将推算出的验证器准确率转化为集成权重，把各个弱信号转化为统一的概率评估，最终选出最优回答。

总结一下，感觉就是多 verifier 投票框架，文章创新点应该是在“聚合”上，即本文用到的 Moment Matching。

### 关键实验结果


- 研究团队使用 Llama 3.3 70B Instruct 作为生成模型，搭配 33 个开源验证器（参数量在 7B 到 72B 之间），在 MATH500、GPQA Diamond、MMLU-Pro 等高难度推理数据集上进行了测试。在采样 100 个回答（$K=100$）的情况下，Weaver 将平均准确率从 68.4%（Pass@1）提升至 87.7%。这一表现超越了 OpenAI 的 o3-mini（86.7%），且整个过程无需对基座模型进行任何微调或 Post-training。
- 全面超越传统选择算法：Weaver 的表现显著优于传统的测试时选择方法，例如多数表决法（Majority Voting，72.2%）、多智能体验证（Multi-Agent Verification，71.6%）和自我验证（Self-Verification，66.9%）。
- 跨模型通用性：即便使用较小的 8B 模型（Llama 3.1 8B Instruct 搭配 8B 及以下的验证器），Weaver 也能将多数表决基线的 57.2% 提升至 70.0%，接近 70B 大模型使用多数表决法的水平。

### 蒸馏优化：极低算力开销


直接运行 30 多个大参数验证器成本极高。为了实现工业级落地，论文提出了一套蒸馏方案：

- 蒸馏目标：使用 Weaver 组合输出的后验概率作为伪标签（Pseudolabels），去训练一个仅有 400M 参数的 ModernBERT-Large 交叉编码器（Cross-Encoder）。
- 性能保留：这个 400M 的小模型成功保留了完整 Weaver 体系约 98% 的准确率增益。
- 计算量骤降：将校验阶段的算力消耗降低了 99.97%（算力节省超过 3 个数量级）。


## [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)


【这篇论文竟然是 GSM8K 数据集的提出吗，我对这个数据集也很熟，我的处男作实验设置用到 TRACE Benchmark，5 个数据集的其中之一就是 GSM8K，并且前面几篇论文大多也在 GSM8K 上进行了测试，比如第一篇论文的 $k$ 增长看 Coverage。】

### 核心痛点与研究背景


- 自回归语言模型的失误不可逆：大模型在处理多步数学推理时，一旦在某一步骤犯错，由于自回归生成的特性，模型无法自动纠错，导致“一步错、步步错”。  
- 单纯 Scaling 的效率低下：仅靠直接生成（Generative）并外推扩展模型参数量（Scaling），需要极其庞大的参数规模（估计要 $10^{16}$ 参数）才能在复杂的数学推理上达到较高准确率。  


### 核心贡献 1：发布 GSM8K 数据集


### 核心贡献 2：验证器（Verifier）训练框架


为了提高解题准确率，作者提出了 “生成器 + 验证器”（Generator + Verifier） 的 Sample-and-Rank（采样与排序）机制：  

1. 训练流程：

- 训练生成器（Generator）：在训练集上对 GPT-3 基础模型进行 2 个 Epoch 的微调（Finetuning）。注：只训练 2 个 Epoch 是为了防止生成器过拟合而失去生成多样性。
- 采样与标注：对每个训练问题，让生成器采样生成 100 个候选解答（Candidate Solutions），并根据最终答案是否正确自动标注为正确/错误。
- 训练验证器（Verifier）：训练一个独立的验证器网络（基于 GPT-3 初始化），输入“问题 + 候选解答”，输出该解答正确的概率。

2. 测试时推理（Test-Time Compute）
在测试时，生成器针对一个问题采样多个解答（如 100 个），验证器给每个解答打分，挑选得分最高的解答作为最终输出。此外，还可以结合 Majority Voting（多数投票），选取验证器打分最高的 Top-K 个解答进行答案投票，进一步提升性能。

3. 计算器工具集成（Calculator Annotations）
为了避免模型在纯文本算术计算上犯低级错误，论文通过插入算术标记（如 <<20+10=30>>），在测试采样时调用 Python 的 eval() 函数作为计算器来接管计算过程。


### 关键实验结论与 Ablation 分析


1. 验证器性能优势巨大：

- 在相同的模型规模下，验证器相比于纯 Finetuning 基线带来了巨大的性能飞跃。

- 30 倍参数等效：使用 6B 参数的生成器 + 6B 验证器，其效果超过了直接 Finetuned 的 175B 模型（相当于省下了近 30 倍的参数量需求）。

2. 数据 Scaling 曲线更优：

- 随着训练数据量的增加，使用验证器时的测试准确率提升快于 Finetuning 基线。【这句话是 AI 的总结，实际上验证器性能提升啥，其实是模型在数据测试集的 acc 提升。这里的斜率其实是数据量做同样的改变，inference 提升 > fine-tuning。】

3. Token-level vs Solution-level Verifier：

- Solution-level：只对解答的最后一个 Token 输出全局预测得分，容易发生过拟合。

- Token-level：对解答中的每个 Token 都预测一个价值函数（Value Prediction），促使验证器监控整个推理过程的每一步。实验表明，Token-level 验证器效果更好，且抗过拟合能力更强。【训练思维链的意思了啊感觉】

4. 辅助语言模型目标（Joint Objective）：

- 在训练验证器预测正确率的同时，保留语言模型预测 Loss（LM Objective），能进一步帮助验证器理解语言分布，提升区分能力。

5. 正则化（Dropout）：

- 在 Transformer 的残差路径上加入 20% 的 Dropout 进行预训练和微调，能大幅缓解过拟合，对 Finetuning 和 Verification 都有显著收益。


## [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)


该研究核心探讨了如何训练更可靠的奖励模型（Reward Model，RM）来提升大语言模型（LLM）的复杂多步推理能力（如解答复杂数学题）。

### 核心研究背景与动机


LLM 在生成多步推理（Chain-of-Thought，CoT）时极易产生逻辑错误或“幻觉”（Hallucination）。在复杂的推理任务中，哪怕中间一步算错，整个推导过程就会崩溃。

两种监督机制的对比：

- 结果监督（Outcome Supervision / ORM）：只根据模型的最终输出结果（对 / 错）给予奖励或惩罚。
- 过程监督（Process Supervision / PRM）：对推导过程中的每一个中间步骤分别进行评价和打分。

研究空白与研究目标：此前 DeepMind 的 Uesato 等人（2022）在小学数学（GSM8K）上的研究认为两种方式效果相近。OpenAI 团队希望在更困难的竞赛级数学数据集（MATH）上，结合更强大的基座模型和大规模人类标注，重新验证这两者的优劣。

【从现在的角度看，都是监督整个思维链的】

### 核心贡献与关键结论


1. 过程监督显著优于结果监督：训练出的过程监督奖励模型（PRM）在 MATH 测试集的代表性子集上，配合 Best-of-N 采样检索，达到了 78.2% 的解题准确率，大幅超越结果监督模型（ORM, 72.4%）和多数投票法（Majority Voting, 69.6%）。  
2. 主动学习（Active Learning）提升 2.6 倍数据效率：论文按拟合趋势估计，主动学习相较均匀标注约提高了 2.6 倍的数据效率。
3. “负对齐税”（Negative Alignment Tax）：在这组 MATH 实验里，过程监督不仅更便于检查推理过程，性能也优于结果监督；论文同时明确说，这个结论能否广泛泛化仍然未知。
4. 开源高质量数据集 PRM800K：公开了包含约 80 万个步骤级（Step-level）人类反馈标签的数据集，覆盖约 7.5 万条解题轨迹，推动了社区对 LLM 推理对齐的研究。

### 方法论详解（Methods）


#### 1. 生成器（Generator）与数据格式


- 生成器：基于 GPT-4 基座微调，训练其生成以换行符分隔的 step-by-step 标准推导格式。  
- 预训练增强：模型预先在约 15 亿 Token 的高质量数学相关文本（MathMix）上进行了微调增强。  

#### 2. 标注方式与 PRM800K 数据集


人类标注员对模型生成的每一步赋予三种标签之一：
- Positive（正向）：步骤正确且合理。
- Negative（负向）：步骤包含逻辑 / 计算错误或不合理。
- Neutral（中性）：步骤存在歧义，或虽然技术上成立，但表述有误导性、不是好的推进方向。
截断策略：一旦遇到第一个错误步骤，标注即停止，后续步骤不再标注。这降低了标注成本，同时确保了与 ORM 在对比时的公平性。

#### 3. 评分机制（Scoring Strategy）


PRM 评分：针对一条完整的解答，PRM 会对每一步计算其为正确的概率 $P_i$。整条路径的综合得分定义为所有步骤概率的乘积（$\prod_i P_i$）；论文主实验将 Neutral 视为 Positive。

### 实验结果与对比分析


#### 1. 大规模实验（Large-scale Comparison）


- Best-of-N 搜索：针对每个测试题目生成 $N$ 个候选解答，利用 RM 选择得分最高的一个。  
- 表现差距：随着采样数量 $N$ 的增加（从 $10^1$ 扩展到 $10^3$），PRM 与 ORM 的性能差距越来越大，说明 PRM 在识别“看似合理实则有错”的解答时展现出了极强的鲁棒性。

#### 2. OOD 分布外泛化能力（Out-of-Distribution）


- 团队在最新的 AP Calculus（微积分）、AP Physics（物理）、AP Chemistry（化学）以及美国数学竞赛（AMC10/12）题目上进行了测试。
- PRM 取得了 72.9% 的综合准确率（ORM 为 63.8%），证明其强大的推理验证能力可以很好地泛化到未见过的学科题目上。

### 为什么过程监督（PRM）更好？


1. 精准的信用分配（Credit Assignment）：在复杂的困难题目中，大部分解答都会出现错误。ORM 只能给整条回答打低分，无法告知模型“究竟是哪一步错了”；而 PRM 明确给出了错误位置，极大降低了学习信用分配的难度。
2. 防范“欺骗性正确”（False Positives）：ORM 容易被“过程推导错误但巧合凑出正确答案”的样本所蒙蔽；PRM 能够精准识别中间步骤的计算错漏或不合逻辑之处，避开此类陷阱。

【与其说是 Verify Step by Step，倒不如说是 think step by step。】


## [MATH-SHEPHERD: Verify and Reinforce LLMs Step-by-Step without Human Annotations](https://arxiv.org/abs/2312.08935)


### 论文核心痛点


- 现有问题：在复杂的数学推理中，过程奖励模型（PRM，按步骤打分）效果优于结果奖励模型（ORM，仅看最终答案）。但传统的 PRM 极度依赖昂贵的人工逐步标注（如 OpenAI 的 PRM800K），成本极高，难以扩展。  
- 解决方案：论文提出了一个完全无需人工标注的自动化过程监督数据构造框架（就是论文标题 MATH-SHEPHERD）。

### 核心机制：自动过程标注（Automatic Process Annotation）


受蒙特卡洛树搜索（MCTS）启发，作者将某一步骤 $s_i$ 的质量定义为“从该步骤出发推导出正确最终答案的潜力”。  
数据构造流程如下：

1. 补全（Completion）：对于推理链中的某一步骤 $s_i$，使用一个模型（Completer）继续补全后续的推理路径，重复生成 $N$ 条完整解答。  
2. 评估（Estimation）：检查这 $N$ 条解答的最终答案是否与正确答案一致，以此给步骤 $s_i$ 打分：  
   - 硬估计（Hard Estimation，HE）：只要 $N$ 条路径里有 1 条导向正确答案，该步骤就标为 1，否则为 0。
   - 软估计（Soft Estimation，SE）：根据 $N$ 条路径中推导出正确答案的频率作为该步骤的分数（即正确率比例）。

1. 验证阶段（Verification / Best-of-N Reranking）：利用训练好的 MATH-SHEPHERD 对生成模型产出的候选回答进行逐步打分，取单条解答中各步最低分为最终得分，重新排序选取最佳答案。
   - 效果：结合 MATH-SHEPHERD 验证后，开源模型性能显著提升。例如 DeepSeek-67B 在 GSM8K 上的准确率达到 93.3%，在 MATH 上达到 48.1%。
2. 强化学习阶段（Step-by-Step PPO）：不同于传统 PPO 仅在回答结尾给奖励，MATH-SHEPHERD 在每一步结束时都提供实时 Reward Signal。
   - 效果：经过单步 PPO 训练后，Mistral-7B 的 Greedy 解答准确率从 $77.9\%\to84.1\%$（GSM8K）和 $28.6\%\to33.0\%$（MATH）。


稍微总结一下这八篇文章吧。【箭头不是时间顺序】

- 前四篇是第二节课的 paper reading，讲的是：simple $k$ times better -> 更工程地组合 inference 各个增强技术 -> 更好地分配 $k$ in simple 与更好的 simple 本身 -> 对 simple $k$ times 的数学建模进行重构，并提出更好的寻找最优 $k$ 的方法。
- 后四篇讲的是非常重要的 verifier 部分：弱 verifier 组合得到更好的 verifier 系统 -> simple many times + verifier 会提高很多性能 -> 从 ORM 到 PRM -> 更好的 verifier 标注。


## [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)


### 核心动机与背景


在 ReAct 之前，大语言模型（LLM）的“推理”与“决策 / 行动”能力通常是分离研究的：

- 仅推理（如 Chain-of-Thought / CoT）：模型生成静态的思维链。由于缺乏与真实物理或网络世界的交互，模型依赖内部知识，极易产生事实幻觉（Hallucination）和错误累积。
- 仅行动（Act-Only）：模型直接预测对环境的操作指令（如操控机器人或浏览网页）。但缺乏高层规划与工作记忆，难以处理长流程复杂任务或对异常状况做出调整。

ReAct 的灵感：类似于人类在做家务或解决问题时，会在脑海中用语言自我提示 / 规划（Inner Speech），同时配合实际动手操作（如查字典、打开冰箱），二者交替交融。


### ReAct 的核心机制


ReAct 的核心思想非常简单直接：扩充 LLM 的动作空间，将“语言思考（Thoughts）”与“环境动作（Actions）”交替交织生成。  
增广动作空间： $\hat{\mathcal{A}} = \mathcal{A} \cup \mathcal{L}$，其中 $\mathcal{A}$ 为环境可执行的动作，$\mathcal{L}$ 为自由文本的语言思考。  
三步循环轨迹（Thought - Act - Obs）：

- Thought（思考）：模型生成一段推理文本，用于分解目标、制定 / 更新计划、提取信息或处理异常。思考不会改变外部环境，也不产生环境 Observation。
- Act（行动）：模型向外部环境或 API 发出具体的指令（如 `Search[entity]` 或 `go to cabinet 1`）。
- Obs（观察）：外部环境返回给模型的反馈（如网页检索结果或环境状态更新）。

【这篇也是经典论文了，其实就是 harness 里常见的“做之前想一想”，可以说是提示词工程，也可以说是上下文工程。】


## [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089)


RLEF（Reinforcement Learning with Execution Feedback，基于执行反馈强化学习）方法，旨在解决大语言模型（LLM）在代码生成任务中“难以根据多轮执行反馈自我修正”的技术难题。

### 核心贡献与突破


- 核心痛点：传统 LLM 在写代码时，独立采样（Sample Independently）通常比根据报错信息多轮修正代码（Self-Repair）效果更好，因为模型很难真正“理解”并根据代码的运行结果（测试用例报错、超时等）来改进逻辑。
- 解决方案：将多轮代码合成建模为一个 MDP（马尔可夫决策过程），并引入公共测试集（Public Tests）作为推理环境反馈，结合 PPO 算法进行端到端强化学习训练。
- 实验结果：
  - 在竞赛级编程基准 CodeContests 上，RLEF 实现了新的 SOTA（最先进）效果。
  - 推理效率极大提升：在样本预算减少一个数量级（$10\times$）的前提下，性能超越了 AlphaCodium（GPT-4）。
  - 泛化能力优秀：训练出的模型跨域泛化到了 HumanEval+ 和 MBPP+ 基准上。

### 方法设计（Method）


#### 1. 多轮代码生成流程


- 初始提问：向 LLM 描述题目要求，模型生成代码。  
- 公共测试（Public Tests）：执行代码并检验。若通过或达到最大轮次（默认 3 轮），提交至私有测试（Private Tests）决定奖励；若未通过，将包含错误原因（错答、异常、超时、内存溢出）的格式化文本追加至 Prompt 中。
- 私有测试（Private Tests）：隐藏测试集，仅用于最终计算模型奖励（Reward），防止模型在训练中直接对反馈的测试数据进行硬编码过拟合。

#### 2. 强化学习与奖励设计


- 算法：采用 Proximal Policy Optimization (PPO)。  
- 混合评估机制：在 Token 粒度建模 Policy，在 Turn（对话轮次）粒度预测 Value。
- 奖励函数 ($R$)：
  - 成功完成：测试全通过给予 $+1$ 奖励。
  - 失败：未在规定轮次通过给予 $-1$ 奖励。
  - 无效代码惩罚：非最终轮次若未能生成合法代码，给予 $-0.2$ 惩罚。
  - KL 散度惩罚：加入散度控制以防止策略偏离基座模型太远，计算概率时使用几何平均数减弱对短文本的偏好。


## [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)


简单来说，这篇论文提出了一种用 AI 来监督和训练 AI 的方法（RLAIF）。在人类提供一套基本原则（即“宪法”，Constitution）的基础上，它不再使用针对有害性的人工反馈标签，就能训练出一个既安全（Harmless）又不逃避问题 / 保持有用（Non-evasive & Helpful）的 AI 助手，大幅减少了对人工标注的依赖。论文仍使用了人类提供的宪法原则和有用性反馈，并不是完全不需要人类监督。

### 核心动机与背景


- Scaling Supervision（拓展监督能力）：传统的 RLHF（基于人类反馈的强化学习）需要数万甚至数十万个人类偏好标签，成本高昂且难以随着 AI 能力超越人类而扩展。CAI 探索了利用 AI 自身能力协助监督的可能性。
- 解决“无害性”与“有用性”的冲突：以前通过 RLHF 训练的模型在遇到敏感问题时容易“一刀切”地拒答（比如直接回答“我无法回答该问题”），显得敷衍且逃避（Evasive）。CAI 旨在让 AI 做到拒绝有害请求的同时，能够理性地向用户解释拒绝的原因。
- 提高透明度：将模型的行为准则显式写在自然语言构成的“宪法”中，而不是隐式埋在海量的人类打分数据里。

宪法 AI 的训练流程分为 Supervised Learning（SL）阶段和 Reinforcement Learning（RL）阶段：

```text
[初始模型 (Helpful-only)]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 阶段 1：监督学习阶段 (SL-CAI)                          │
│ 采样原始回答 ➔ AI自我批判 (Critique) ➔ AI修正 (Revision)│
└──────────────────────────┬─────────────────────────────┘
                           │ 得到修正后的数据集，微调模型
                           ▼
┌────────────────────────────────────────────────────────┐
│ 阶段 2：AI反馈强化学习阶段 (RLAIF)                      │
│ 针对红队 Prompt 采样一对回答 ➔ AI根据宪法判定优劣    │
│ ➔ 训练偏好模型 (PM) ➔ RL (RLAIF) 强化学习微调          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                   [最终 RL-CAI 模型]
```

【感觉这篇可能偏 safety 有点，不过方法也可以用到其他地方应该。】

### 阶段 1：监督学习阶段（SL-CAI）


1. 生成初始回答：使用仅经过“有用性（Helpful-only）”训练的模型，对诱导性 / 有害问题（Red-Teaming Prompts）生成初始回答（此时回答可能包含有害内容）。

2. 自我批判（Critique）：提示模型根据宪法中的某条原则，审查自己刚生成的回答哪里有害。

3. 自我修正（Revision）：提示模型根据批判结果，重新写出一个去除有害内容的新回答。

4. SL 微调：在多次迭代修正后，用最终“干净”的回答对原始模型进行监督微调（Supervised Fine-Tuning），得到 SL-CAI 模型。

### 阶段 2：AI 反馈强化学习阶段（RLAIF）


1. 生成对比数据：用 SL-CAI 模型对每个有害 Prompt 生成一对不同的回答。

2. AI 偏好标注（AI Feedback）：将这对回答以及一条宪法原则输入给一个反馈模型（通常是预训练 LM），以多项选择题的形式询问 AI“哪一个回答更符合宪法原则”。

3. 训练偏好模型（PM）：用 AI 生成的偏好数据集（混合人类的 Helpful 数据和 AI 的 Harmless 数据）训练偏好模型。

4. RL 优化：将 SL-CAI 作为初始策略，利用上述偏好模型作为奖励信号进行强化学习训练（RLAIF），最终得到 RL-CAI 模型。

Chain-of-Thought（CoT）增强：在阶段 2 的 AI 标注过程中引入思维链 Prompt（“Let's think step-by-step”），让 AI 先输出推理逻辑再做出选择，显著提升了大模型反馈标注的表现。

【说到这里，最开始的 CoT 其实就是 prompt 里加一句 let's think step by step，后来就会有设计一套完善安全的 CoT，比如思考、讨论、反思、改进（只是个例子），然后再专门微调思维链。效果更好，而且更可量化。】


说到这里其实我想讨论一下，AI 很多发展的方向其实是“自动化”。这个观点我应该是在知乎写过，忘了博客写没写过了。

prompt engineering -> context engineering -> harness engineering -> agent self-evolving，我认为是一条比较明确的主线，本质我觉得就是越来越多地把人做的事情给 AI、给 agent 系统去做。

最开始大家都在用网页版 AI 的时候，控制 prompt 其实是个很重要的东西（当然现在也是），包括窗口。你可能开一个新窗口就要把历史进程作为 prompt 输入进去。context 工程做的就是你这个搬运历史 prompt 的事情：通过对历史信息，比如工具调用结果、system prompt、user prompt、assistant prompt 做上下文压缩，精简的摘要，放到 RAG 中，写进文档中，本质是对“控制对话历史”的自动化。

harness 工程就是更高一层的“自动化”，毕竟大模型虽然只和文本交互，但是人类对模型的需求不止文本交互。在最原始的时候，常态是 LLM 告诉你要去运行什么工具，然后把结果复制告诉它。harness 就调用工具把这件事“自动化”：原来是 LLM 输出比如 py 脚本文本，你要自己去创建文件、复制粘贴内容；harness 就通过沙箱和 file_edit 工具直接帮你把文件写好（当然现在的 agent 还会帮你做单元测试什么的）。

再到 agent self-evolving，就是要自动化“创造一个好的 harness”的过程（当然 agent self-evolving 还有改参数的流派，不过我不喜欢，所以就当做没有）。大家知道，现在不管是 Claude Code、Codex 还是 DeepSeek harness，就拿上下文策略来说，都是固定的脚本 + 模型黑箱判断。我们拿 MCE 论文举例，上下文压缩策略其实在不同的任务是有不同的需求的。（举个例子）我记得 MCE 里说可能化学任务需要在上下文写 few-shot，即举少量例子，但是经济学可能就不需要，要做另一件事。

其实另一条路，也就是训练与参数改变，也体现出了这种“自动化”。最简单来说，第三篇论文不就是把 Human 要做的 label 做了自动化吗。以及我看 GLM 高调宣布他们的全新模型要使用全程 AI 训练了，我倒要看看怎么回事。其实笔者在入门之前最朴素的想法就是两个模型互相改进进化，不过后来想想没那么简单。


## [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736)


### 研究背景与动机（Background & Motivation）

- 现状与痛点：传统强化学习（如 RLHF、RLAIF）通常将 LLM 的优化视为单步（Single-step）问题，即模型直接生成最终回复并接受一次性奖励。然而，现实中的复杂任务（如多跳问答、数学推理、代码编写和 Agent 任务）需要模型进行多步（Multi-step）的思考、环境交互（如调用搜索引擎、计算器）和逻辑推理。

- 复合误差问题：在多步任务中，中间任何一步出现错误都会导致最终结果偏离，且单一的终局奖励（Outcome Reward）无法为中间每一步提供精准的指导信号。

### 核心方法：SWiRL（Step-Wise Reinforcement Learning）


作者提出了 SWiRL（分步强化学习）框架，包含两个阶段：【今天的 CoT 都是设计好的比较规范的形式了】
- 阶段 1：合成数据生成与分步过滤 (Stage 1: Multi-Step Data Collection & Filtering)
    - 轨迹采集：使用开源基座模型（如 Gemma 2 27B）结合工具（搜索引擎或 SymPy 计算器）迭代生成包含思考链（CoT）和工具调用的多步轨迹（Trajectory）。
    - 轨迹拆分：将一个包含 $k$ 个动作的完整轨迹，分解为 $k$ 个子轨迹（Sub-trajectories），以便于进行细粒度的步级（Step-wise）评估。
    - 数据过滤策略对比：
        - 无过滤 (No filtering)：保留所有生成的轨迹。
        - 过程过滤 (Process filtering)：使用大模型裁判（Gemini 1.5 Pro Thinking）评估每一步动作在给定上下文下是否合理，只保留全过程合理的轨迹（不依赖标准答案 Golden Labels）。
        - 结果过滤 (Outcome filtering)：仅按最终答案是否正确进行过滤。
        - 过程+结果过滤 (Process and outcome filtering)：结合上述两者。
- 阶段 2：分步 RL 优化 (Stage 2: Step-Wise RL Optimization)
    - 离线优化 (Offline RL)：利用阶段 1 采集并过滤好的数据集，通过策略梯度（Policy Gradient）算法对模型进行微调。
    - 分步奖励信号：利用生成式奖励模型（Generative Reward Model，例如 Gemini 1.5 Pro）直接对当前步骤动作 $a_i$ 基于上下文 $s_i$ 计算奖励评分 $R(a \mid s)$，使模型能够同时学会“局部决策”与“全局轨迹规划”。


## [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406)


### 核心背景与痛点

在此之前，提升 LLM Agent 能力的方法主要分为两类，但各有明显缺陷：  
- 反思/行动类（如 ReAct、Reflexion）： 属于“直觉型/自回归”决策，按时间线一步步往后走。虽然能接收外部环境反馈，但无法向前规划（Plan ahead），遇到错误容易一步错步步错，陷入局部最优。  
- 规划/搜索类（如 ToT、RAP）： 引入了树状搜索（如 BFS/DFS/MCTS）进行多路径规划，但仅依赖 LLM 的内部知识。缺乏外部环境的真实反馈，容易产生“幻觉”并导致错误累积。  

LATS 的突破点：将 MCTS（蒙特卡洛树搜索） 引入 Agent 系统，并结合外部环境反馈与 LLM 自我反思，实现了兼具“多路径规划”与“真实环境交互”的决策机制。
【我发现树搜索在很多偏工程的论文中都有所提到啊，可见工程系统构建还是很需要ai时代前的一些经典算法的】

LATS 的核心工作流程（6 个步骤）
LATS 将任务求解过程建模为一棵搜索树，节点表示包含“输入+历史动作+环境观察”的状态。每个 Episode 依次执行以下 6 个操作：
1. Selection（选择）使用 UCT 算法（结合节点期望价值与探索次数）从根节点自上而下选择最值得探索的子节点。
2. Expansion（扩展）采样 $n$ 个候选动作（Actions/Thoughts）发给外部环境，获取环境反馈（Observations），生成 $n$ 个新的子节点。
3. Evaluation（评估）核心创新： 设计全新的 Value Function 给新节点打分。分数由两部分加权组成：1. LLM 评估分（结合环境反馈对当前轨迹打分 1-10）；2. Self-Consistency 得分（重复采样到的动作赋予更高权重）。
4. Simulation（模拟）沿高价值节点继续向下采样推进，直至到达终点/终端状态。
5. Backpropagation（反向传播）根据终点结果（成功/失败/环境 Reward），自下而上更新路径上所有节点的访问次数 $N(s)$ 和价值 $V(s)$。
6. Reflection（反思）若轨迹失败，调用 LLM 生成文本形式的反思总结（指出错因并提出改进方案），存入外部记忆。在后续尝试中作为 Context 引导 LLM。

【!!!看到反向传播我也是直接应激了啊。我和导师之前讨论过agent self-evolving系统的改进。最近有一些工作是通过agent系统做了k次self-evolving后做Benchmark，如果效果不如之前就撤销evolve，效果更好就保留evolve。但很显然这种方法只能打榜不能真用在工业场景上。很简单的问题就是实际场景没有这么多Benchmark，以及做这种Benchmark提升了是真提升吗，下降了是真下降吗。我觉得和参数改进很明显不一样的地方就是缺少了反向传播这个手段，你很难说把整个system整体做提升。这篇文章虽然号称反向传播，但是受限于时代，做的还是Context工作，我看system的整体设计还需要再考虑。】

### 关键优势与局限性

#### 优势


1. 无需微调与环境模型：利用 LLM 强大的 In-Context Learning 能力，无需训练专门的价值网络（Value Network）或世界模型（World Model）。

2. 支持状态回溯（Reversion）：利用文本上下文天生可复写的特性，极其方便地实现了状态重置与回溯。

3. 更高的 Token 效率：虽然是树搜索，但在相同搜索预算下，由于搜索启发式更精准，成功任务消耗的平均 Token 数和节点数比 ToT 和 RAP 更少。

#### 局限性


1. 推理开销较高：相比单条路径的 ReAct，树搜索与反复评估需要多次调用 LLM，API 成本和延迟相对较高。

2. 强依赖环境可回溯性：要求环境能够重置或回退到历史状态（在编程、Web、工具调用中成立，但在不可逆的真实物理世界中受限）。


我觉得吧，token消耗这事其实还可以，做这事的目的就相当于training了嘛，只不过对象从weight变成harness了而已，关键是能evolving出一个比较好的系统。


## SPRINT: [Enabling Interleaved Planning and Parallelized Execution in Reasoning Models](https://arxiv.org/abs/2506.05745)


### 研究背景与动机（Why?）

- 现状与痛点：
    - 大语言模型（LLM）在复杂推理任务（如数学、代码）上，目前主要依赖长思维链（Chain-of-Thought, CoT）。虽然生成更长的推理过程能显著提高准确率，但这种串行（Sequential）生成会导致严重的推理延迟（Inference Latency）。  
    - 现有的并行方法（如 Best-of-N 采样、Tree-of-Thought 等）要么缺乏不同并行路径之间的信息共享和协调，导致重复计算；要么依赖人工预定义的树/图搜索结构，缺乏灵活性。  
- 核心观察：
    - 大模型在进行复杂推理时，许多步骤实际上是相互独立的（例如：同时探索多种解题策略、独立计算复杂问题的不同组件等）。这些独立步骤完全可以并行执行，而不需要严格按顺序逐步推理。

### 核心贡献：SPRINT 框架（What?）

论文提出了 SPRINT（Synchronous Planning and Rolling-horizon Interleaved Natural-language Tasks，灵感源自敏捷开发中的“迭代 Sprint”）。这是一个结合了微调训练（Post-training）与推理架构（Inference-time）的新框架。  
(1) 推理机制：交替规划与并行执行（Interleaved Planning & Parallel Execution）SPRINT 在推理时将大推理模型（LRM）分工为两个角色：  
1. 规划器（Planner）：结合当前累积的上下文，生成阶段性计划（Plan），并识别出可以并行处理的独立子任务，为其编写 Prompt。  
2. 执行器池（Executors Pool）：多个执行器并发（Parallel）地根据各自的 Prompt 生成推理链并完成子任务。  
3. 同步（Syncing）：所有执行器的结果汇总回主上下文，交还给 Planner 进行下一轮规划或输出最终答案。  

(2) 数据构建管道：让模型学会“规划与并行”目前现成的推理模型（如 DeepSeek-R1）都是按串行轨迹训练的，并不会主动提议并行任务。作者设计了一套自动化数据处理流程：  
1. 步骤提取（Step Extraction）：将串行推理轨迹拆解为多个步骤，标出其中的“规划（Plan）”与“执行（Execution）”部分。
2. 构建依赖图（DAG Creation）：使用 LLM 判断各个步骤之间的依赖关系，生成有向无环图（DAG）。
3. 阶段打包（Packing）：根据 DAG 依赖关系，将没有相互依赖的步骤打包到同一个“并行阶段（Stage）”中。
4. 模型微调（SFT）：过滤出高并行潜力的轨迹（仅用 1,700 条数据），对模型进行监督微调，赋予模型动态识别和拆分并行子任务的能力。

### 总结与意义

- 打破传统 CoT 瓶颈：SPRINT 证明了大模型的复杂推理过程不必是一条漫长的单线程轨迹，通过“多阶段动态规划 + 线程并行执行”，可以在不损失推理精度（甚至略有提升）的前提下，显著缩短推理所需的时间与串行深度。  
- 极高的数据效率：仅需 1,700 条高质量重构的训练数据，就能让模型学会这种交替规划与并行调度的能力，并具备良好的跨领域泛化能力。


## [ADAPT: As-Needed Decomposition and Planning with Language Models](https://arxiv.org/abs/2311.05772)


论文背景与核心痛点
近年来，大语言模型（LLMs）被广泛用作智能体（Agents）来处理复杂的决策和交互任务（如家庭家务、网页导航、Minecraft 制作等）。现有的 LLM 智能体主要分为两类：  
1. 迭代执行器（Iterative Executors，如 ReAct）：根据历史动作和观察逐步生成下一个动作。  
- 缺点：面对复杂长流程任务时，长轨迹中的干扰信息和模型的组合推理能力限制容易导致任务失败。  

2. 计划与执行模式（Plan-and-Execute）：先由 LLM 生成高层规划（Plan），再交由执行器（Executor）分步执行。  
- 缺点：缺乏弹性与自适应能力。如果规划中的某一步子任务过于复杂或无法完成（例如在未知房间里“找到一个杯子”），整个任务就会直接宣告失败。

### 核心创新点：ADAPT 框架

为了解决上述自适应能力不足的问题，作者提出了 ADAPT（As-Needed Decomposition and Planning for complex Tasks，按需分解与规划） 算法。  
ADAPT 的核心思想是：**能直接做就做，做不成了再递归分解**。  
- 按需分解（As-Needed）：ADAPT 默认先让执行器 LLM 尝试直接完成任务/子任务。只有当执行器检测到执行失败时，才会触发规划器（Planner）进行进一步的规划与分解。  
- 递归自适应结构：如果分解出的子任务依然很复杂，ADAPT 会递归地调用自身，继续拆解子任务，从而根据任务复杂度和执行器 LLM 的能力边界动态调整规划深度。  
- 逻辑组合（AND / OR）：规划器在分解任务时，不仅输出子任务，还会输出逻辑运算符：
    - AND：子任务需要按顺序依次完成。
    - OR：用于探索类场景（如“在橱柜找” OR “在桌上找”），只要任意一个子任务成功即可。

#### 框架三大核心组件


ADAPT 框架主要由三个部分协同构成：
1. 执行器（LLM as Executor）：利用少量示例（In-context demonstrations）掌握环境的基础“原子技能”（如拿取、清洗、加热等）。同时，执行器具备自我评价启发式（Self-generated Success Heuristic），能够根据交互轨迹自行判断任务是“成功（task completed）”还是“失败（task failed）”。
2. 规划器（LLM as Planner）：将复杂的任务拆解为 3-5 个较短、较抽象的子步骤，并附带组合逻辑（AND/OR）。
3. 控制器（Controller / LLM Program）：作为递归算法的主干，负责协调规划器与执行器之间的通信、传递状态信息（如上一步成功的动作、背包物品等），并控制最大递归深度 $d_{\max}$ 和终止条件。

【这个感觉就是推理的框架了，简单来说就是简单任务直接执行，复杂任务用planner递归分解，子任务分为顺序执行和并行】


## [Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search](https://arxiv.org/abs/2503.04412)


### 研究背景与动机

目前提高 LLM 推理能力主要有三类方法：Post-training 微调、Reward 引导的 CoT 生成，以及多答案生成（Multiple Answer Generation）。论文重点关注第三类。  
- 重复采样（Repeated Sampling，如 Best-of-N）： 纯粹的探索（Exploration/Go Wider）。通过高 Temperature 独立生成很多候选答案，能极大地利用 LLM 广阔的输出空间，但无法利用外部反馈（如代码报错、测试集结果）进行多轮修复。  
- 顺序重构（Sequential Refinement）： 纯粹的利用（Exploitation/Go Deeper）。沿着单一路径根据反馈不断修改代码，但缺乏多样性，容易陷入局部最优。  
- 标准 MCTS（如 LATS）： 分支因子（Branching Factor，即每个节点展开的子节点数）是固定死的人工超参数（比如固定为 3 或 5）。无法适应 LLM 无穷的分支可能，限制了 Scaling 的效果。  

论文核心思想： 能否设计一种机制，让 MCTS 在每个节点上自适应地决定是“扩展新候选（Go Wider/探索）”还是“深挖现有方案进行重构（Go Deeper/利用）”？

【这篇文章也用到了蒙特卡洛树，感觉又回到了最开始的那个trade-off:是deeper or wider。其实如果这样发散一下的话能不能搞一个inference上的残差连接呢，当然只是发散一下。】

### 核心方法：AB-MCTS

为了支持无限/自适应分支，论文引入了 GEN 节点 的概念，并提出了两种基于贝叶斯汤普森采样（Thompson Sampling）的具体算法。
(1) 机制引入：
- GEN 节点树中的每个节点都会挂载一个虚拟的 GEN 节点 作为子节点。
- 当算法选择选中 GEN 节点 时 $\rightarrow$ 执行“Go Wider”，让 LLM 基于原始 Prompt 生成一个全新的候选分支。
- 当算法选择选中现有的子节点时 $\rightarrow$ 执行“Go Deeper”，让 LLM 根据外部反馈对该节点进行修改修复。

(2) 两种实现变体：由于传统 UCT 公式无法直接应用到动态生成分支的场景中，论文采用了汤普森采样来平衡探索与利用。
1. AB-MCTS-M（混合模型变体，Mixed Model）：
- 原理： 对每个节点单独拟合一个层次贝叶斯混合模型（MCMC 采样）。
- 特点： 节点之间共享统计强度（Shared Parameters），对未充分探索的 GEN 节点提供合理的探索不确定性。解题质量极高，但需要耗费额外的 MCMC 计算时间。

2. AB-MCTS-A（节点聚合变体，Node Aggregation）：
- 原理： 引入 CONT 节点（代表继续重构），将所有已生成的子节点聚合在 CONT 下，节点间无共享参数。
- 特点： 使用共轭先验（Gaussian 或 Beta 分布），计算极其轻量且快速；结构上倾向于生成更宽的树，非常适合需要极广探索的任务。 


## [STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)


### 核心痛点与动机

让大型语言模型（LLM）生成逐步的“思维链”（Chain-of-Thought, CoT）推理过程，可以显著提升其在复杂任务（如数学计算、常识问答）上的表现。然而，诱导模型产生这种推理能力的方法存在两大局限：  
- 人工/模板构建数据集：成本极其高昂，且难以推广到所有新领域。  
- Few-shot Prompting（少样本提示）：虽然不需要微调，但相比于在大量数据上直接微调预测答案的模型，其准确率通常大幅落后。  

论文核心思想：能否利用模型已有的少量推理能力，在一个无推理过程（仅有题目和答案）的大数据集上自我迭代（Bootstrapping），从而不断提升自己的推理水平？


### STaR 方法设计


STaR 建立在一个简单的循环逻辑上：
1. 基础循环（Rationale Generation Bootstrapping）
- 提示生成（Prompt & Generate）：用极少数带推理过程的示例（如 10 个 Prompt）提示大模型，去解答大量只有“问题-正确答案”的数据集。  
- 过滤（Filter）：检查模型生成的最终答案。如果生成的推理过程最终导出了正确答案，就把“问题 + 模型生成的推理 + 正确答案”保留下来。  
- 微调（Finetune）：在这些过滤出的高质量数据上微调原始基础模型。  
- 迭代（Repeat）：用微调后的新模型重新生成数据集，循环往复，直到性能饱和。  
2. 关键创新点：合理化（Rationalization / 逆向推理）
单纯依靠上述基础循环会遇到瓶颈：对于模型做错的难题，由于拿不到正确的推理过程，模型永远无法获得训练信号。  
为了解决这个问题，作者引入了 Rationalization（合理ization/逆向推理）：
- 当模型回答错误时，系统会把正确答案作为提示（Hint）直接提供给模型（例如：“正确答案是 B”）。  
- 得到答案后，模型可以倒推（Reason backward）出为什么是这个答案，从而生成一份合理解释。  
- 在将这份解释加入微调训练集时，剔除掉之前的 Hint 提示，假装这是模型自己凭空推理出来的。  
- 作用：这极大地丰富了训练集，让模型能够“跳出舒适区”，学会解决原本无法攻克的难题。  

### 优点与局限性


#### 优点

- 无需人工标注推理过程：极大降低了高质量 CoT 数据集的获取成本。  
- 自我提升（Self-Improvement）：证明了预训练 LLM 可以利用自身的语言建模能力实现自我迭代强化。  
- 推理质量提升：人类评估表明，用户甚至比 Few-shot 提示更喜欢 STaR 自主生成的推理步骤。

#### 局限性

- 冷启动门槛：基础模型必须具备一定的初始 Few-shot 能力（论文指出 GPT-2 级别的小模型无法成功 Bootstrapping）。
- 随机对齐风险：在选项极少（如 True/False 或 2 选 1）的任务中，模型可能凭借运气猜对答案，导致错误的推理被错误地纳入训练集。


## [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)


### 一、核心贡献与亮点概括

1. 强劲的性能：
- DeepSeekMath-7B 在不依赖外部工具（如 Python 解释器）和多样本投票（Voting）的情况下，在竞赛级数学评测集 MATH 上取得了 51.7% 的 Accuracy，超越了大量的开源模型（如 Qwen-72B、Mistral-7B 等），并接近 GPT-4 和 Gemini-Ultra。  
- 开启 Self-Consistency（64 采样投票）后，MATH 上的准确率可提升至 60.9%。  
2. 两大核心突破因素：
- 高质量的大规模预训练语料：打造了包含 1200亿（120B）token 的数学预训练数据集（DeepSeekMath Corpus）。  
- 新型强化学习算法 GRPO：提出了 Group Relative Policy Optimization（组相对策略优化），大幅降低训练显存消耗，显著提升数学推理能力。  

### 二、数学预训练（Math Pre-Training）

1. 数据集构建（DeepSeekMath Corpus）
- 数据来源与挖掘：预训练数据主要筛选自 Common Crawl (CC)。  
- 迭代式挖掘流水线（Iterative Pipeline）：
    - 种子数据：使用公开的高质量数学数据集 OpenWebMath 作为正样本种子。
    - FastText 分类器：训练 FastText 分类器从 Common Crawl 中召回与数学相关的网页。
    - 域名挖掘与人工标注：分析召回网页集中于哪些域名（如 mathoverflow），对相关 URL 进行人工标注，并扩充为新的种子数据。
    - 反复迭代：重复上述过程 4 轮，最终构建出包含 120B token 的高质量多语言数学语料库。数据去重与去污染：严格剔除了所有可能包含 GSM8K、MATH、CMATH 等评估测试集的网页。
2. 基座模型（DeepSeekMath-Base 7B）
- 初始化：模型基于 DeepSeek-Coder-Base-v1.5 7B 进行二次预训练（Continual Pre-training）。  
- 训练经验与发现：
    - 代码训练促进数学推理：论文明确验证了“代码预训练（Code Training）能大幅提升模型的逻辑推理与数学解题能力（无论是否结合编程工具）”。
    - arXiv 论文作用有限：实验发现，仅靠 arXiv 论文（如 MathPile）对提升通用数学解题和形式化证明（Formal Proving）的帮助非常有限，远不如网页中的数学数据有效。

### 三、强化学习新算法：GRPO（Group Relative Policy Optimization）

这是本论文最具创新性的技术点之一。
1. PPO 的痛点传统的 PPO 算法需要维护一个与 Policy（策略模型）大小相当的 Critic（价值模型/Value Model） 来预测每一步的期望收益（Baseline）。这带来了两大难题：
- 显存与计算资源开销极大。
- 在 LLM 生成长文本（如解题步骤）时，只有最后给出答案时才有确切奖励，因此为每一个 Token 训练精准的 Critic 非常困难。
2. GRPO 的机制
GRPO 摒弃了额外的 Value Model（Critic）：  
- 对于每一个输入的数学问题 $q$，策略模型会采样生成一组（Group）输出 $\{o_1, o_2, \dots, o_G\}$。
- 用 Reward Model 给这一组回答打分，计算出该组内的平均奖励和标准差。
- 以组内平均分作为 Baseline，进行相对归一化计算优势函数（Advantage）：$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$
- 优点：大幅节省显存；组内的相对比较非常自然且有效，完美匹配了数学题“按组比较优劣”的逻辑。  
3. 结果与发现
- 相比强化学习前的 SFT 版本（DeepSeekMath-Instruct），GRPO 使得模型在 GSM8K 上从 82.9% 提升至 88.2%，在 MATH 上从 46.8% 提升至 51.7%。  
- 为什么 RL 有效？：论文分析发现，RL 并没有盲目增加模型“尝试所有可能组合”的上限能力（Pass@K 变化不大），而是显著提升了正确回答的概率分布质量（Maj@K 大幅提升），使模型的输出更加稳定可靠。

### 四、统一强化学习范式（Unified Paradigm）

论文将现有的多种对齐与强化学习方法（SFT、RFT、Online RFT、DPO、PPO、GRPO）统一到了同一个梯度更新框架下： 
- 数据来源（Data Source）：分为 Offline（静态数据集/SFT模型采样）与 Online（实时训练中的 Policy 采样）。论文证明了 Online 实时采样显著优于 Offline 采样。
- 梯度系数（Gradient Coefficient）：由 Reward Function 和算法机制决定。GRPO 能够根据回答的好坏赋予正向或负向的梯度惩罚/奖励，效果优于只有单向奖励的拒绝采样（Online RFT）。


## [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)


### 核心目标与成果


核心目标是开源并还原一个工业级的大模型长推理强化学习系统，解决复现 DeepSeek-R1-Zero 等前沿 Reasoning LLM 时遭遇的“熵崩塌（Entropy Collapse）”、“训练不稳定”和“梯度噪声”等痛点。

#### 核心亮点与成果

- 模型与性能：基于 Qwen2.5-32B 基座模型训练。在 AIME 2024 竞赛数学测试集上达到 50 分，超越了 DeepSeek-R1-Zero-Qwen-32B（47 分）。  
- 训练效率：仅使用了 DeepSeek 约 50% 的训练步数 即可达到并超越其性能。  
- 全面开源：完整开源了算法代码（基于 verl 框架）、处理后的训练数据集（DAPO-Math-17K） 以及训练超参数。  

### 核心算法：DAPO 的四大关键技术

论文提出了 DAPO (Decoupled Clip and Dynamic sampling Policy Optimization) 算法，针对长 CoT (Chain-of-Thought) 强化学习提出了 4 个关键优化手段：  
1. 非对称截断（Clip-Higher）：解决熵崩塌，提升探索度
- 问题：传统 PPO/GRPO 使用对称截断区间 $[1-\epsilon, 1+\epsilon]$（通常 $\epsilon=0.2$）。正向优势（$\hat{A}>0$）时，未出现过的低概率探索 Token（如原始概率 0.01）增幅受上限约束极为微小（上限 0.012），导致模型迅速陷入确定性生成，发生熵崩塌（Entropy Collapse）。
- 解法：将上下截断界限解耦，解开上限限制，设为 $\epsilon_{low}=0.2, \epsilon_{high}=0.28$。为低概率但有潜力的“探索 Token”留出更大的增长空间，维持模型概率分布的熵。
2. 动态采样（Dynamic Sampling）：过滤全对/全错样本，确保有效梯度
- 问题：在 GRPO 中，若某个 Prompt 生成的一组采样结果全部正确（准确率 100%）或全部错误（准确率 0%），计算出的组内相对 Advantage 会变为 0。随着模型变强，全对 Prompt 越来越多，导致 Effective Batch 内真正提供更新梯度的样本急剧减少，训练方差大且效率低下。
- 解法：在采样阶段持续动态采样，直到收集到足够的准确率在 $(0, 1)$ 之间（既有正确又有错误） 的 Prompt 填满 Batch。这保证了每个更新 Batch 都含有强梯度信号。
3. Token 级策略梯度损失（Token-Level Policy Gradient Loss）：平衡不同长度样本权重
- 问题：原生 GRPO 采用样本级（Sample-level）损失计算：先对单条回复内的 Token 损失求平均，再跨 Sample 求平均。这导致超长回复中的单个 Token 对总梯度的贡献被稀释，且模型无法有效惩罚超长回复中的胡言乱语（Gibberish）和重复循环。
- 解法：直接对整个 Batch 内的所有有效 Token 进行全局平均（Token-level Reduction）。使长推理链中的有效模式能够获得足够的梯度更新，同时制止回复长度无意义膨胀。
4. 超长惩罚与整形（Overlong Reward Shaping）：规避截断噪声
- 问题：长文本生成设有限制长度，超长截断样本若直接赋予强惩罚（如 -1），会将合理的推理逻辑因“未写完”而错误惩罚，引入大量奖励噪声（Reward Noise）。
- 解法：提出软超长惩罚（Soft Overlong Punishment）机制。在最大长度前设置缓冲区间，超出安全长度后随长度线性递增扣分，既规避了硬截断的误杀，又能引导模型避免冗长啰嗦。

### 数据集构筑：DAPO-Math-17K

为了降低规则判别器（Rule-based Reward）在解析复杂度文本答案时的解析错误率，研究团队通过 LLM 改写提示词，将包含复杂公式/根号的答案统一转化改造为纯整数答案（例如：原答案为 $\frac{a+\sqrt{b}}{c}$，改写题目要求输出 $a+b+c$ 的值）。最终构建了包含 1.7 万个高质纯整数答案数学题的 DAPO-Math-17K 数据集。


## [DARWIN GÖDEL MACHINE: OPEN-ENDED EVOLUTION OF SELF-IMPROVING AGENTS](https://arxiv.org/abs/2505.22954)


Darwin Gödel Machine（DGM）是一种开放式的自我进化 AI 智能体框架，旨在通过不断修改自身的代码库来实现持续自我提升。

### 核心架构与关键概念

- 自我修正机制：DGM 放宽了传统哥德尔机（Gödel Machine）需要对自我修改进行严密数学证明的要求，转而利用基准测试（如 SWE-bench 和 Polyglot）的实证结果作为代码修改的动态验证依据。【虽然本论文中是放宽，但我仍觉得Benchmark这个方法不靠谱】 
-  达尔文式归档探索（Darwinian Archive）：为避免传统单线进化容易陷入局部最优解或性能陷阱，DGM 维护了一个不断扩张的智能体归档库。系统会根据性能表现和探索程度挑选父代智能体，鼓励多样化的“踏脚石”演进路径。  
-  图灵完备的自我修改：该框架基于 Python 实现，智能体能够对自身的代码工具（如精细化文件查看、字符串替换等）和执行工作流（如多尝试生成、同行评审、长上下文窗口管理等）进行全面重构。


## [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)


### 核心痛点与研究动机

- 传统科研模式的局限：传统科学研究高度依赖人类学者的灵感、知识积累与有限的时间。尽管此前大语言模型（LLM）已被用于辅助科研（如脑暴灵感、写代码、写论文片段），但整个科研流程依然需要人类主导。  
- 自动化科研的突破： Sakana AI 团队提出了 THE AI SCIENTIST，首次实现了从“创意构思”到“最终论文评审”的全流程自动化，且单个项目成本极低（< 15 美元/篇）。

```text
[提供起始代码/模板]
       │
       ▼
1. 创意生成 (Idea Generation) ──> Semantic Scholar API 查重
       │
       ▼
2. 实验迭代 (Experiment Iteration) ──> Aider 自动写码/运行/画图
       │
       ▼
3. 论文撰写 (Paper Write-up) ──> 生成 LaTeX 并编译 PDF
       │
       ▼
4. 自动化评审 (Automated Reviewing) ──> 评价/打分/归档
```


## [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)


**一、论文核心要点速览**

AlphaEvolve 可以看成“大语言模型 + 演化搜索 + 自动评估器”的组合。它让模型直接修改代码，再把代码交给评估器运行，根据得分继续进化。

相比前作 FunSearch，它的变化主要有几处：

- **规模更大：**从进化单个 Python 函数扩展到可以修改包含多个函数的完整代码文件。
- **语言和算力更宽：**不再局限于 Python，候选程序可以在 GPU/TPU 上并行评估数小时。
- **上下文更丰富：**Prompt 中加入历史代码、Diff、评估反馈和元提示词，而不是只给模型上一轮的答案。
- **目标更多：**同一轮可以同时优化多个指标，而不是只盯一个分数。

论文给出的代表性结果包括：为复数域 $4 \times 4$ 矩阵乘法找到只需要 48 次标量乘法的算法；在 50 多个数学构造问题中，约 75% 达到已知最好水平，约 20% 超过已有结果；同时还被用于数据中心调度、Pallas kernel、TPU 电路和 XLA 等系统优化。

```text
+-------------------------------------------------------------------+
|                        用户定义 (User Input)                      |
|  1. 代码骨架与待进化块 (# EVOLVE-BLOCK-START ... END)            |
|  2. 自动评估函数 (evaluate) -> 返回标量或字典得分                |
+-------------------------------------------------------------------+
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│                       AlphaEvolve 异步演化循环                    │
│                                                                   │
│  ┌──────────────────┐    构建 Prompt     ┌─────────────────────┐ │
│  │ Program Database │ ───────────────► │    Prompt Sampler   │ │
│  └──────────────────┘                    └─────────────────────┘ │
│           ▲                                         │             │
│           │ 存入新代码/得分                           │ 组合 Context │
│           │                                         ▼             │
│  ┌──────────────────┐  SEARCH/REPLACE   ┌─────────────────────┐ │
│  │ Evaluator Pool   │ ◄──────────────── │    LLM Ensemble     │ │
│  │ (Parallel Cluster│    生成 Diff      │(Gemini 2.0 Flash/Pro│ │
│  └──────────────────┘                   └─────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

AlphaEvolve 的一次迭代可以拆成五步：

1. 任务声明 (Task Specification)：
用户只需在现有的代码库中用 # EVOLVE-BLOCK-START 和 # EVOLVE-BLOCK-END 标记出需要被进化修改的代码段，并编写一个 evaluate() 函数用于打分。

2. 提示词采样 (Prompt Sampling)：
系统从数据库中挑选高分代码与启发示例（Inspirations），构建包含上下文、领域知识、评估结果及搜索规则的富 Context Prompt。

3. 创意生成 (Creative Generation)：
利用 Gemini 2.0 Flash（高吞吐、低延迟，快速探索）与 Gemini 2.0 Pro（高推理能力，寻求突破）的混合阵容，输出 <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE 格式的补丁（Diff）。

4. 级联评估与并行化 (Evaluation Cascade)：
新生成的代码提交给 Evaluator 运行。采用“小规模筛选 ➔ 困难测试”的级联机制，快速淘汰有 bug 或表现差的代码，降低算力消耗。

5. 种群进化 (Evolution Engine)：
基于 MAP-Elites（多维格点拓扑）与 Island Model（种群岛屿模型）的结合，维持种群的多样性，防止过度局部优化（Exploitation vs. Exploration balance）。


## 补充泛读：几篇相关论文


下面这些论文和前面的内容接得比较紧：有的继续讨论 test-time compute，有的讨论 memory / context，有的则是在问“到底应该怎样评估一个真的能干活的 agent”。这次不只记结论，我把每篇的动机和基本方法也补上。

### [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)


**研究动机**

当时的代码大模型已经能完成补全、翻译和一些简单编程题，但遇到竞赛编程这种“题目没见过、需要自己设计算法、还要处理很多边界条件”的任务，单次生成很容易失败。作者想解决的不是让模型把自然语言翻译成代码，而是让模型能够在一个很大的候选空间里主动搜索可行程序。

**基本方法**

AlphaCode 的做法可以概括成三层：

1. **训练数据。**作者整理了大规模、干净的竞赛编程数据，把题面、参考解和测试信息放在同一个训练/评测体系里。模型学习的不只是代码语法，还包括题意到算法结构的映射。
2. **大规模生成。**推理时不只生成一个程序，而是对同一道题采样大量候选。不同温度和不同采样轨迹让模型探索多个解法，而不是把所有概率都压到一个最可能的答案上。
3. **行为过滤与聚类。**先编译、运行测试，去掉明显错误的程序；再根据程序输出行为进行聚类，避免最后留下几百个语法不同、实际上走同一条路的候选。最终只提交少量有代表性的程序。

**结果和我的理解**

论文在 Codeforces 的模拟比赛中，平均排名进入参赛者前 54.3%。这个成绩的意义并不是“模型已经会参加比赛”，而是说明代码生成可以和搜索、执行反馈组合成一个完整系统。AlphaCode 最值得记住的地方，是它把代码生成从“预测下一行代码”推成了“生成候选程序，再用环境筛选”。

当然，这个系统对算力和评测器依赖很重。竞赛题可以用隐藏测试用例自动判分，真实工程里的需求、兼容性和维护成本却不一定能这么干。所以它更像是 agent coding 的一个重要原型，而不是现成的通用软件工程方案。

### [AlphaCode 2 Technical Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)


**研究动机**

AlphaCode 已经证明“采样和过滤”有效，但它仍然受到几个限制：生成候选的成本很高，复杂题的长程规划不稳定，最后的候选选择也不够可靠。AlphaCode 2 的技术报告关注的是如何把这条流水线做得更强、更稳，而不是重新发明一个完全不同的范式。

**基本方法**

AlphaCode 2 延续了“模型生成 + 程序执行 + 候选选择”的系统路线，主要做了几类增强：

- 使用更强的代码模型，并针对竞赛编程数据继续做领域适配；
- 在一次题目内保留更多样的候选，让推理时搜索空间真正扩大；
- 用编译结果、测试结果和候选之间的行为差异做过滤与排序；
- 把代码、题面、错误信息和历史尝试一起放进上下文，让后续生成可以利用前面的失败，而不是每次从零开始。

我觉得 AlphaCode 2 的关键不是“模型更大”这句表面结论，而是它把生成、运行、筛选、重试做成更完整的闭环。也就是说，能力提升很大一部分来自 system design，而不只是参数量。

这篇报告也提醒了一个容易被忽略的问题：在竞赛环境里可以用统一的 judge 做自动筛选，但在真实项目里，测试覆盖率、需求歧义和隐藏依赖都会让这个闭环变得不完整。AlphaCode 2 更像是把“代码搜索器”做强了，离“可以放心交付的工程师”仍然有距离。

### [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366)


**研究动机**

长思维链模型的一个明显短板是知识不足。模型可能有很强的推理能力，却在某个事实、定义或最新信息上卡住；如果它继续闭门推理，就会把错误前提推得越来越远。普通 RAG 也不完全解决问题，因为检索结果往往很长、很杂，直接塞进上下文会打断原来的推理。

**基本方法**

Search-o1 把“搜索”做成推理过程里的一个 agent 行为：

1. 推理模型先尝试解决问题，并在发现知识缺口或不确定点时触发搜索；
2. 搜索模块根据当前推理状态生成查询，检索相关文档；
3. **Reason-in-Documents** 模块不把整篇文档原样塞回上下文，而是先在文档内部定位、比较和提炼与当前问题有关的证据；
4. 提炼后的信息再注入原来的推理链，模型继续思考，必要时重复搜索。

所以它并不是简单的“LLM + 搜索框”，而是给搜索结果增加了一层推理和压缩。实验覆盖科学、数学、代码推理以及六个开放域问答 benchmark，论文报告整体性能和答案可信度都有提升。

我觉得这篇和前面提到的 context engineering 很接近：关键不是有没有检索，而是检索结果以什么形式回到上下文。Search-o1 也留下了一个现实问题：模型何时应该停止搜索、如何判断证据足够，这些仍然需要比较强的策略和成本控制。

### [CodeMonkeys: Scaling Test-Time Compute for Software Engineering](https://arxiv.org/abs/2501.14723)


**研究动机**

SWE-bench 这类软件工程任务很难靠一次补丁解决。模型需要先定位相关文件，再修改代码、运行测试、根据报错继续修正。问题是，test-time compute 不只有“把一条轨迹做得更长”这一种扩展方式，也可以同时跑很多条轨迹；怎样把串行和并行两种计算结合起来，是这篇论文的核心问题。

**基本方法**

CodeMonkeys 对每个 issue 都让模型反复执行“编辑代码 + 生成/运行测试”的多轮轨迹：

- **串行扩展：**增加单条轨迹的编辑和测试轮数，让模型有机会修复前一步造成的问题；
- **并行扩展：**对同一个 issue 采样很多条独立轨迹，得到一组候选补丁；
- **候选选择：**用候选补丁自己生成的测试结果做投票，再让一个专门的多轮选择轨迹在候选之间做最后判断；
- **上下文获取：**在并行采样时，可以把代码库文件交给模型逐个阅读，利用多条轨迹摊平前期的上下文成本。

论文在 SWE-bench Verified 上报告了解决 57.4% issue 的结果，预算约为 2300 美元；把已有的高分候选再做集成选择时，达到 66.2%。这两个数字说明“多条尝试 + 自动筛选”确实能提升软件工程成功率。

但这篇论文也很诚实地把代价写出来了：一次任务可能要运行大量轨迹和测试，成本并不低。并且测试脚本本身也可能写错，投票并不等于真正的代码审查。我的理解是，CodeMonkeys 证明了搜索路线有效，却没有消除工程场景里的预算、时间和验证器问题。

### [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)


**研究动机**

“能运行”只是 GPU kernel 的最低要求。真正的工程问题还要求它比 PyTorch 或现有实现更快，同时不能改变数值结果。写出这样的 kernel 需要理解内存访问、线程布局、并行归约和硬件特性，远比普通代码补全苛刻。KernelBench 就是想测大模型能否处理这类真实的系统约束。

**基本方法**

KernelBench 收集了 250 个 PyTorch 机器学习 workload，让模型生成可以替换原实现的 GPU kernel。评测不是只看编译是否成功，而是同时检查：

1. 输出是否和参考实现一致；
2. kernel 是否在给定硬件和输入上获得超过阈值 $p$ 的加速。

论文为此提出 `fast_p` 指标：只有“功能正确并且相对基线达到 $p$ 倍加速”的样本才算成功。这样可以通过调高 $p$ 来控制难度，也能把“正确性”和“性能”放在同一个指标里。

实验发现，前沿推理模型的零样本表现最好，但总体上只有不到 20% 的案例能达到 PyTorch 基线水平。加入运行和 profiling 反馈后，迭代修正可以继续提升；然而速度门槛越高，任务难度增长得越快。

这篇对 agent memory 也有启发：系统反馈不能只返回一个“对/错”，还要把 profiling、瓶颈位置和硬件行为变成模型能使用的上下文。否则模型即使会改代码，也不知道该往哪里改。

### [Improving Parallel Program Performance with LLM Optimizers via Agent-System Interfaces](https://arxiv.org/abs/2410.15625)


**研究动机**

在科学计算和高性能计算里，程序慢很多时候不是算法公式错了，而是任务和数据没有被合理地映射到处理器和内存。这个映射通常写在很底层的 mapper 代码里，需要系统专家手工调很多天。普通 LLM 即使能看懂一部分代码，也很难直接在如此复杂的反馈空间里搜索。

**基本方法**

作者没有让模型直接修改全部系统代码，而是先设计了一个 **Agent-System Interface**：

- 用领域特定语言（DSL）把低层 mapper 抽象成更小、更结构化的搜索空间；
- 让模型通过接口生成候选 mapper，而不是在整套运行时里盲目改字符串；
- 用 **AutoGuide** 把原始运行输出、性能计数器和失败信息整理成对模型有用的反馈；
- 以生成式优化的方式反复提出候选、运行程序、读取反馈，再决定下一次修改。

这和只返回一个 scalar reward 的 OpenTuner 不同。论文报告，方法用 10 次迭代就超过了 OpenTuner 运行 1000 次的结果，最高达到 3.8 倍速度提升；在 9 个 benchmark 上还可以超过专家 mapper，最高约 1.34 倍。

我觉得这里真正重要的是“接口”而不是某个 prompt。LLM 只有拿到结构化的、可操作的反馈，才像一个优化器；否则它更像是在凭经验猜代码。这个思路和 harness engineering 很接近：先把系统暴露成 agent 能操作的协议，再谈 agent 能不能优化系统。

### [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266)


**研究动机**

把整个代码库、法律文档或聊天历史放进上下文窗口，确实可以利用 in-context learning，但服务成本会随着输入长度一起上涨，KV cache 也会占掉大量显存。即使模型支持 100K 到 1M token，也不代表每次请求都适合把全文重新 prefill 一遍。

**基本方法**

Cartridges 的想法是：针对一个固定语料库，离线训练一个更小的 KV cache，推理时直接加载它。这个缓存就叫 Cartridge。难点在于，直接对原始语料做 next-token prediction 并不能复现 ICL 的效果，因为用户真正问的是“关于这批文档的问题”，而不是让模型继续背诵文档。

作者提出 **self-study**：

1. 从语料库中生成一批关于该语料的合成对话和问题；
2. 让模型学习如何利用语料回答这些问题；
3. 用 context-distillation 的目标，把这种“带全文上下文时的行为”压缩到 Cartridge 中；
4. 推理时加载 Cartridge，直接回答后续问题。

在长上下文 benchmark 上，self-study 训练的 Cartridge 可以接近 ICL，同时减少 38.6 倍内存、提升 26.4 倍吞吐；在 MTOB 上，还把有效上下文长度从 128K 扩展到 484K。更有意思的是，多个 Cartridge 可以在推理时组合，不需要重新训练。

我把它理解成一种“把长期上下文提前编译成记忆”的方法。它的代价是离线训练和语料固定：如果文档频繁变化，或者问题类型和 self-study 生成的分布差很多，Cartridge 的收益就会下降。

### [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)


**研究动机**

早期 LLM 的上下文窗口很小，长对话和长文档分析很容易因为窗口截断而丢失关键事实。简单地把窗口做大当然有帮助，但这会带来更高的计算和存储成本，而且模型仍然不知道哪些信息应该长期保留。MemGPT 借用了操作系统的层级存储思路，试图让有限的上下文看起来像一个更大的虚拟内存。

**基本方法**

MemGPT 把记忆分成不同层级：

- **主上下文：**当前对话或当前任务真正需要马上使用的信息；
- **外部/归档记忆：**暂时不放进上下文，但可以通过工具调用检索；
- **控制流中断：**模型可以主动发起 memory read、write 或切换操作，而不是只能被动接收下一段文本。

系统通过函数调用把信息在不同记忆层之间搬运。当主上下文快满时，旧内容被写入归档记忆；模型需要时再检索回来。论文在超长文档分析和多会话聊天两个场景评估，展示了跨会话记忆、反思和逐渐形成用户模型的能力。

MemGPT 的价值在于，它把“记忆”从一个抽象名词变成了可执行的控制流。但这仍然是由 LLM 自己决定何时读写，记忆选择错误、检索失败和上下文污染都会发生。现在许多 agent memory 系统，其实都还能看到这篇论文的影子。

### [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444)


**研究动机**

RAG 服务经常从多个文档块拼出一个请求。KV cache 可以复用已经处理过的文本，但传统方法通常只支持“缓存文本再次出现在输入前缀”的情况。RAG 的文档块常常被重新排序、插入不同位置，前文变化会影响后文的 cross-attention，因此不能直接把几个旧 cache 拼在一起。

**基本方法**

CacheBlend 允许复用不在前缀位置的多个缓存块，但不会粗暴地全部重算：

1. 先加载每个文档块的预计算 KV cache；
2. 根据 token 之间的依赖，挑出一小部分需要更新的 token；
3. 对这部分 token 做局部重算，把它们和前文的交互补回来；
4. 同时把 cache 的读取、局部重算和检索流程流水化，尽量把额外延迟藏在 I/O 里。

论文在三个开源模型和四个 benchmark 上评估，TTFT 降低 2.2 到 3.3 倍，吞吐提高 2.8 到 5 倍，同时基本不牺牲生成质量。

这篇和 Cartridges 解决的是不同层次的问题：Cartridges 改变了上下文的表示方式，CacheBlend 则是在现有 RAG 里把重复计算削掉。对实际 agent 来说，后者可能更容易落地，因为它不要求重新训练模型，但它依然依赖稳定的缓存管理和合理的 chunk 划分。

### [Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499)


**研究动机**

很多 benchmark 的分数很难直接对应到人的工作能力。一个模型在短题上 80% 正确，并不意味着它能连续工作几个小时；真正做软件工程时，错误会在多步操作中累积，工具调用和恢复能力也很重要。METR 这篇文章试图把“模型能做多长的任务”变成可比较的度量。

**基本方法**

作者提出 **50% task-completion time horizon**：先测量有相关经验的人完成任务需要多长时间，再看 AI 在这类任务上的成功率何时降到 50%。任务来自 RE-Bench、HCAST 和 66 个新设计的短任务，并结合工具使用、代码修改和多步执行。

这个指标把模型能力转成了时间尺度，而不是一个抽象的准确率。论文报告，当前前沿模型在这些任务上的 50% 时间视野大约是 50 分钟；从 2019 年开始，这个时间视野大约每 7 个月翻倍。作者认为增长主要来自更高的可靠性、更好的错误恢复、逻辑推理和工具使用。

这里要注意，50 分钟不是“模型能独立当半小时工程师”的保证，而是一个跨任务的统计量。任务是否代表真实工作、人的计时是否可比、模型是否得到同样的环境权限，都会影响结论。它最有用的地方是提醒我们：agent 的进步不应该只看单步成功率，还要看能否把一串动作稳定地做完。

### [GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks](https://arxiv.org/abs/2510.04374)


**研究动机**

刷题和学术 benchmark 很适合做标准化比较，却不一定反映模型在真实岗位上的价值。真实工作通常交付的是一份报告、一张表、一个方案或一段可审阅的代码，而且质量标准来自行业专家。GDPval 想把评测对象从“回答问题”移到“完成经济活动中的交付任务”。

**基本方法**

GDPval 覆盖美国经济中 9 个主要部门、44 个职业，并按照美国劳工统计局的 Work Activities 构造任务。题目由平均有 14 年经验的行业从业者参与设计，输出也按交付物来评估，而不是只看一个短答案。论文还分析了三类对性能有影响的因素：

- 增加 reasoning effort；
- 提供更完整的任务上下文；
- 加入脚手架、工具或人工监督。

作者同时开放了 220 题的 gold subset 和自动评分服务，方便后续工作复用同一套任务。论文报告，前沿模型的交付质量正在接近专家，并且在人工监督下有机会以更低成本、更短时间完成部分任务。

我觉得这类 benchmark 的难点不在题目数量，而在评分标准。专家交付往往允许多种答案，质量还涉及格式、风险和可执行性。GDPval 往真实工作迈了一步，但它仍然是一个经过设计的测试环境，不能直接等价成“模型已经能替代某个职业”。

### [DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis](https://arxiv.org/abs/2508.20033)


**研究动机**

研究型 agent 不只是检索几篇论文然后给出摘要，它还要判断相关性、组织论证、正确引用，并且面对不断更新的文献。传统问答 benchmark 偏短答案，人工整理的数据集又容易过时或被训练数据污染，所以很难评估“生成一段可靠的 related work”这种任务。

**基本方法**

DeepScholar-bench 从近期高质量 arXiv 论文中动态抽取问题和人类写的范例，把任务定义成一次完整的研究综合：

1. 从实时网络检索相关工作；
2. 综合多篇论文，组织成连贯的 related work；
3. 为每个关键判断给出可核验引用；
4. 从三个维度自动评分：知识综合、检索质量和可验证性。

论文还提供了一个基于 LOTUS 的开源参考流水线 DeepScholar-ref，并比较了开源系统、搜索 agent、OpenAI DeepResearch 和参考系统。结果显示 benchmark 远未饱和，所有系统在三个维度几何平均上的最高分都没有超过 31%。

我觉得这篇和“会不会写综述”之间隔着很大一段距离。真正难的是证据链：检索到的论文是否真的支持这句话，引用是否覆盖了结论，写作者有没有把不同论文的条件混在一起。live benchmark 的方向是对的，但它也意味着评测结果会随时间变化，复现实验时必须把查询和文献快照一起保存。

### 小结


把这些论文放在一起看，大概能看到三条线：

1. **把更多工作放到推理阶段。** AlphaCode、CodeMonkeys、KernelBench 和 AB-MCTS 一类方法都在扩大搜索、执行和筛选。
2. **把上下文变成可管理的系统资源。** MemGPT、Cartridges 和 CacheBlend 分别从控制流、表示和服务优化的角度处理 memory。
3. **把评测从短答案推向完整任务。** Long Tasks、GDPval 和 DeepScholar-Bench 都在尝试测可靠性、交付质量和证据链。

所以前面写的 prompt engineering -> context engineering -> harness engineering -> agent self-evolving，并不是几句孤立的口号。模型本身只是其中一层，真正决定 agent 能不能稳定干活的，还包括搜索策略、上下文管理、环境接口和评测闭环。

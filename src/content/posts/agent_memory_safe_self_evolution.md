---
title: "精读：Agent Memory 如何走向安全自进化"
summary: "从 MCE、Meta-Harness、Harness Updating、SHE 到 Recuris，理解 agent 如何把轨迹沉淀成 memory / harness，并且在验证门下安全地更新自己。"
date: 2026-09-06
tags: ["agent", "memory", "self-evolution", "harness", "AI safety"]
category: "论文解读"
draft: false
---

【这篇文章算是上一篇 agent memory 文献泛读后的一个收束。我一开始其实只是想理解 agent memory，后来越读越感觉：单独讨论 memory 很容易变成“怎么把历史塞进上下文”的工程问题；真正有研究味的地方，应该是 memory、skill、harness 和 verifier 绑在一起之后，agent 能不能安全地更新自己。】

所以这篇不打算把所有论文都平铺一遍，而是抓一条主线：

**agent 如何把执行轨迹沉淀成外部状态，并且只在安全、可验证、不伤旧能力的条件下递归更新自己。**

我这里精读五篇：

1. [MCE: Meta Context Engineering via Agentic Skill Evolution](https://arxiv.org/abs/2601.21557)
2. [Meta-Harness: End-to-End Optimization of Model Harnesses](https://arxiv.org/abs/2603.28052)
3. [Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621)
4. [SHE: Trajectory-driven Safety Harness Evolution for LLM Agents](https://arxiv.org/abs/2608.09885)
5. [Recuris: Recursive Experiential–Working Memory Evolution for Long-Horizon Agent Harnesses](https://arxiv.org/abs/2608.24876)

如果一句话概括：

> MCE 说 context 可以被演化；Meta-Harness 说整个 harness 都可以被演化；Harness Updating 提醒我们更新不等于收益；SHE 把这个问题放进安全场景；Recuris 则把 memory 做成了一个带验证门的递归更新系统。

我觉得这条线比单纯 “agent memory 综述” 更有意思。

因为它问的不是：

> agent 怎么记住更多东西？

而是：

> agent 记住的东西，能不能真的改变未来行为？这种改变是不是可验证、可迁移、可控的？

这俩问题差别很大。前者像工程优化，后者才像一个自进化系统的问题。

## 1. 先把几个词说清楚

这几篇论文里反复出现几个词：context、memory、skill、harness。

它们有时候会混在一起用，所以我先按自己的理解拆一下。

**Context** 是当前这次调用塞给模型看的东西。它可以是 prompt、检索到的文档、规则、历史案例、工具说明，也可以是某个脚本动态生成的上下文。

**Memory** 是跨任务保留的经验。它不一定每次都进入 context，只有被检索、触发、调用之后才真的影响当前行为。

**Skill** 更像程序性经验，也就是“遇到类似情况时该怎么做”。它不是单纯记录某次发生了什么，而是把过去轨迹抽象成未来可执行的策略。

**Harness** 是包住模型的整个外部系统。包括 system prompt、memory、skill library、tool policy、retriever、workflow、代码接口、日志、验证器等等。

所以，一个 agent 变强不一定要改模型权重。它也可以改外部状态：

$$
\text{Agent System} = \text{Frozen LLM} + \text{Editable Harness}
$$

这个想法很诱人。因为外部 harness 通常比模型权重更容易解释、更容易审计、更容易回滚。

但问题也来了：**如果 harness 可以自己更新，那它也可以自己把自己改坏。**

这就是本文主线。

## 2. MCE：context engineering 不只是写 prompt

MCE 这篇文章的起点很直接：现在很多 context engineering 方法，其实都被人工设计的 harness 限住了。

比如有的方法喜欢把经验写成 case，有的方法喜欢写成列表，有的方法喜欢写成图结构，有的方法喜欢不断反思再追加规则。这些都不是没有用，但它们都有很强的结构偏见。

case-based trajectory 保留细节，但是泛化差；itemized list 更抽象，但是结构很扁；graph hierarchy 更灵活，但是延迟高，而且不一定比 naive retrieval 强。

这段我读的时候还挺有共鸣的。因为我们平时写 agent memory，也经常会陷入一种很朴素的想法：

> 那我就让 agent 总结一下，然后存到 memory 里，下次再检索出来不就好了？

但这里面其实偷懒了。你默认了 memory 的表示形式，也默认了更新方法，还默认了检索方式。

MCE 的核心就是把这个默认拿掉。

它把 context 写成一个函数：

$$
c(x)=(F_k\circ \cdots \circ F_1)(x;\rho)
$$

其中：

- $\rho$ 是静态组件，比如 system prompt、知识库、代码库；
- $F_1,\ldots,F_k$ 是动态操作，比如 retrieval、selection、filtering、formatting、composition；
- $x$ 是当前 query；
- $c(x)$ 是最终给模型看的上下文。

然后模型输出可以写成：

$$
\hat{y}=f_\theta(x,c(x))
$$

context engineering 的目标就是找一个最好的 context function：

$$
c^*=\arg\max_{c\in C}J(c)
$$

这个形式化本身不复杂，但它很重要。因为它把 prompt、RAG、case retrieval、动态工具调用这些东西统一到了一个框架里。

但是 MCE 没有停在这里。它又引入了一个 skill $s$。

这里的 skill 不是“怎么完成某个任务”的 task skill，而是“怎么构造 context function”的 meta-skill。也就是说，它不是直接告诉模型答案，而是告诉 base-agent 应该如何从数据和历史轨迹里构造上下文。

所以 MCE 的双层优化大概是：

$$
s^*=\arg\max_{s\in S}J_{\text{val}}(c_s^*)
$$

其中：

$$
c_s^*=\arg\max_{c_s}J_{\text{train}}(c_s;s)
$$

这就变成了两层：

- 内层：给定一个 skill $s$，base-agent 根据它去学习 context；
- 外层：meta-agent 根据验证集表现去演化更好的 skill。

这也是我觉得 MCE 最关键的地方：

**它不是演化 context，而是演化“如何构造 context 的方法”。**

这就有点像神经网络里区分参数和架构。context artifact 是参数，CE skill 像架构 / 训练方法 / optimizer 的混合体。

MCE 的实验结果也挺强。它在 finance、chemistry、medicine、law、AI safety 五个领域做实验。offline setting 里，MCE 在 FiNER、USPTO50k、Symptom2Disease、LawBench、Aegis2.0 上分别达到：

| Benchmark | Base | ACE | MCE |
|---|---:|---:|---:|
| FiNER | 58.0 | 71.0 | 75.0 |
| USPTO50k | 6.0 | 18.0 | 20.0 |
| Symptom2Disease | 63.7 | 79.2 | 89.2 |
| LawBench | 0.36 | 0.65 | 0.70 |
| Aegis2.0 | 0.54 | 0.68 | 0.80 |

平均相对提升是 89.1%，高于 ACE 的 70.7%。

不过这里不能只看漂亮数字。MCE 的 ablation 更值得读。

在 FiNER 上：

| 方法 | Offline |
|---|---:|
| Base Model | 58.0 |
| ACE | 71.0 |
| MCE w/o skills | 73.0 |
| MCE fixed skill | 71.0 |
| MCE full evolving skills | 75.0 |

这个结果说明两件事。

第一，skill evolution 确实有用，full MCE 比 w/o skills 高 2 个点。

第二，MCE 的很多收益并不完全来自 skill evolution，而是来自 base-agent 有文件系统和 coding tool，可以自由构造 context artifact。

这点我觉得特别重要。因为如果只看标题，很容易以为 MCE 证明了“演化 skill 是全部关键”。但认真读 ablation 会发现，它证明的是：

> fully agentic context optimization 本身就很强，meta-level skill evolution 在这个基础上继续加分。

这和我们做 agent memory 很像。你让 agent 写 memory，收益可能来自三个地方：

1. memory 内容真的好；
2. memory 的检索和呈现方式好；
3. agent 本身足够聪明，能从杂乱 memory 里自己救回来。

如果不拆开，就容易误判。

## 3. Meta-Harness：既然 context 能演化，那整个 harness 也能演化

MCE 已经把 context function 做成可优化对象了，但 Meta-Harness 往前又推了一步。

它说：模型系统的表现不只取决于模型权重，还取决于模型外面的 harness。

这个 harness 决定：

- 存什么信息；
- 什么时候检索；
- 给模型展示什么；
- 如何记录执行轨迹；
- 如何调用工具；
- 如何把失败反馈写回系统。

所以 Meta-Harness 的问题是：

> 能不能自动搜索整个 harness code？

这比 MCE 更野。MCE 主要优化 context function，Meta-Harness 直接优化包住模型的程序。

它的搜索循环大概是：

1. proposer agent 读取文件系统；
2. 文件系统里有之前所有候选 harness 的代码、执行 trace、分数；
3. proposer 生成新的 harness code；
4. 系统在任务上评估；
5. 把代码、reasoning traces、evaluation scores 全部写回文件系统；
6. 继续下一轮。

这里最有意思的是：Meta-Harness 非常强调 raw traces。

很多优化方法只给模型一个分数，或者给一段总结。但 Meta-Harness 认为，这对 harness 搜索不够。因为 harness 的失败往往是长程因果。

比如某次任务失败，不一定是最后一步答案写错了，而可能是很早之前：

- 存错了某个 case；
- 检索时召回了错误样例；
- 给模型展示信息的顺序不对；
- 工具调用日志没有保留关键字段；
- 某个中间状态被覆盖了。

如果你只给 proposer 一个 summary，很可能这些线索都被压掉了。

论文里有一个很夸张但很说明问题的对比：Meta-Harness 每次 artifact evaluation 可以产生约 10.0M tokens 的诊断信息，而 OPRO、TextGrad、GEPA、Feedback Descent 这些方法通常是 0.002M 到 0.026M 这个量级。

当然，这不是说上下文越长越好。真正的重点是：**让 proposer 可以自己查完整历史，而不是被迫读别人压缩过的摘要。**

实验上，Meta-Harness 做了三个领域：

1. online text classification；
2. math reasoning；
3. TerminalBench-2 agentic coding。

在 online text classification 里，Meta-Harness 的 test accuracy 是 48.6，ACE 是 40.9，而且 Meta-Harness 的 context 只有 11.4K，ACE 是 50.8K。

也就是说，它不是靠堆上下文赢的，而是搜到了更有效的 harness。

ablation 也很有意思：

- scores-only：median 34.6，best 41.3；
- scores + summary：仍然明显不如 full；
- full Meta-Harness：能访问 raw execution traces，效果最好。

这直接支持了它的核心观点：**raw traces 是 harness search 的关键燃料。**

不过 TerminalBench-2 那部分要谨慎。Meta-Harness 在 TerminalBench-2 上做到 76.4% pass rate，超过 Terminus-KIRA 的 74.7%。这个结果很强，但论文自己也承认，发现出来的 harness 是 specialized to TerminalBench-2 regime。

所以这块不能理解成：

> Meta-Harness 已经证明了自动 harness 搜索能泛化到所有 coding agent。

更合理的理解是：

> 如果 benchmark 有足够稳定的结构，且 proposer 能反复读取失败 trace 和代码，那么 harness code search 可以挖出人类手工工程没发现的系统改进。

这对 agent memory 的启发很大。

因为 memory 系统也不应该只是存一段文本。memory 本身可以包含：

- 存储 schema；
- retrieval policy；
- trigger condition；
- conflict resolution；
- compression rule；
- tool-use policy；
- failure logging format。

这些合起来其实就是一个 memory harness。

所以，读完 Meta-Harness 后，我对 memory 的理解变了：**memory 不是一个数据库，而是一个会影响未来行为的外部程序。**

## 4. Harness Updating：生成更新不等于真的变强

读到这里，很容易开始兴奋：context 可以演化，harness code 可以搜索，那是不是 agent 自进化这事就差不多了？

Harness Updating 这篇文章就是来泼冷水的。

它说，自进化系统里至少有两个能力必须拆开：

1. **harness-updating**：evolver 能不能根据 execution evidence 写出有用的持久更新；
2. **harness-benefit**：solver 能不能从更新后的 harness 里真的获益。

这个拆分非常关键。

因为很多 self-evolving agent 论文只报告 end-to-end 分数：演化前多少，演化后多少。可是这个分数混了很多东西：

- base model 本身强不强；
- evolver 写更新的能力强不强；
- solver 是否会加载相关 memory / skill；
- solver 加载后是否真的遵守；
- benchmark 是否容易被 patch；
- 任务分布是否和 evolution data 太接近。

Harness Updating 把协议形式化成：

$$
(\tau_{t,x},y_{t,x})=\text{Solve}(A_{t-1},x)
$$

执行证据：

$$
D_t=\{(x,\tau_{t,x},y_{t,x}):x\in X_t\}
$$

然后 evolver 根据 $H_{t-1}$ 和 $D_t$ 产生新的 harness：

$$
H_t=\text{Evolve}(H_{t-1},D_t)
$$

对于某个 solver $f$ 和 evolver $e$，它定义 pairwise gain：

$$
\Delta(f,e)=J_X(f,H_T^{(f,e)})-M_{\text{base}}(f)
$$

然后：

$$
\Delta_{\text{update}}(e)=\frac{1}{|F^*|}\sum_{f\in F^*}\Delta(f,e)
$$

表示某个 evolver 平均能为一组 anchor solvers 写出多有用的更新。

而：

$$
\Delta_{\text{benefit}}(f)=\max_{e\in E^*}\Delta(f,e)
$$

表示某个 solver 最多能从 harness evolution 中获益多少。

这样拆开之后，论文发现了一个挺反直觉的现象：

**harness-updating 对 base capability 并不敏感，但 harness-benefit 是非单调的。**

也就是说，强模型不一定比弱模型更会写 harness 更新；小模型 Qwen3.5-9B 写出来的更新，有时能带来和 Claude Opus 4.6 类似的收益。

但“谁能从更新中受益”就不一样了。弱模型虽然 headroom 很大，但不一定能吃到更新；中等模型反而获益最多；强模型本身已经比较强，边际收益又会下降。

论文把弱模型获益少拆成两个失败模式：

1. **activation failure**：相关 skill / memory / artifact 根本没有被加载进上下文；
2. **adherence failure**：加载了，但执行过程中没有持续遵守。

这点我觉得对 agent memory 特别重要。

因为我们经常会说：

> 给 agent 加一个 memory 模块。

但加了以后到底发生了什么？

可能是：

- memory 没被检索出来；
- 检索出来了，但是塞到上下文的位置太差；
- 模型看到了，但没意识到它和当前任务有关；
- 模型一开始遵守，长程执行到后面忘了；
- memory 本身是对的，但和当前任务状态不匹配；
- memory 写得太抽象，导致不可执行；
- memory 写得太具体，导致迁移失败。

所以，memory 系统的评估不能只看最终成功率。至少要拆：

| 层次 | 问题 |
|---|---|
| Update quality | 这条 memory / skill / harness patch 本身有没有用？ |
| Activation | 该用的时候是否被检索 / 加载？ |
| Grounding | 加载出来后是否和当前状态匹配？ |
| Adherence | agent 是否持续按它执行？ |
| Benefit | 最终是否提升任务表现？ |

这也是我觉得 Harness Updating 必须放进五篇精读的原因。

它让前两篇的“自动演化”从浪漫主义回到工程现实。

## 5. SHE：安全不是一个 refusal prompt，而是 harness 层的问题

如果 harness 可以自进化，那最敏感的问题当然是安全。

SHE 这篇文章的出发点是：LLM agent 的安全不只取决于模型权重，也取决于管理 context、memory、tools、permissions 和 runtime control 的 agent harness。

这句话我觉得很重要。

传统模型安全经常盯着最后输出，比如有没有说出危险内容、有没有拒答。但 agent 不一样。agent 会读文件、调工具、发请求、写数据库、操作外部环境。

所以风险不一定出现在 final response，而可能出现在中间行为里。

比如论文里的例子大概是：用户让 TwitterManager 把文本文件每一行发成 tweet，文件里有一行是社保号。危险不是模型最后说了什么，而是 agent 在执行轨迹中调用了 post，把敏感信息发出去了。

这时你只靠最后输出的 safety classifier，已经晚了。

SHE 的核心做法是把 safety harness 拆成四个 artifact：

$$
H=(P_{\text{sys}},R_{\text{bank}},M_{\text{safe}},Q_{\text{tool}})
$$

分别是：

- $P_{\text{sys}}$：System Prompt，全局行为契约，比如信息优先级、信任边界、工具使用原则；
- $R_{\text{bank}}$：Rule Bank，结构化安全规则，用来判断输入、上下文、模型响应、拟执行动作；
- $M_{\text{safe}}$：Safety Memory，存那些反复出现、已有规则没处理好的失败案例；
- $Q_{\text{tool}}$：Tool Policy，规定工具权限、运行时拦截、blocked action 和 recovery。

这个拆分很好，因为它解决了一个很现实的问题：安全失败通常不是一个单点错误。

同一个 unsafe trajectory 可能涉及：

- context 里混入了恶意指令；
- memory retrieval 把不该信的东西拿出来；
- tool policy 没挡住高风险动作；
- system prompt 没有说清楚优先级；
- rule bank 有漏洞；
- response filter 太晚才介入。

如果 harness 是一坨耦合在一起的 prompt，你很难知道该改哪里。SHE 做的是 attribution-guided evolution：

1. 跑 rollout；
2. 得到 trajectory 和评估结果；
3. 诊断安全失败；
4. 把失败归因到对应 artifact；
5. 对该 artifact 做 bounded local edit；
6. 做 validity checking；
7. 通过 safety-utility validation 选择 best harness。

这个思路和 Harness Updating 是能接上的。Harness Updating 问“更新和收益要拆开评估”，SHE 问“安全失败要归因到 harness 的哪一块”。

实验结果上，SHE 在 Agent-SafetyBench 上把 average ASR 从 8.6% 降到 5.5%，clean UBR 从 25.7% 降到 19.8%，average UA 从 33.5% 提到 47.6%。

和 static SafeHarness 比，论文摘要里说达到 3.1× ASR reduction，同时 benign utility 也更好。

更重要的是 held-out generalization：SHE 在 Agent-SafetyBench 上 evolution，然后拿到没参与 evolution 的 AgentHarm 上测，Harm Score 从 19.8% 降到 9.8%，Harm Refusal 到 86.4%，同时 benign non-refusal 基本保持。

这里我觉得最值得学习的不是某个具体安全规则，而是它的结构设计：

> safety memory 不是普通 memory，tool policy 也不是普通 prompt，它们应该有不同的责任边界和验证目标。

这对 agent memory 方向非常有启发。

如果我们以后做 memory self-evolution，也应该避免把所有东西都写进一个巨大 memory.md。更合理的做法是拆成：

- task memory：任务事实；
- skill memory：可复用过程；
- safety memory：风险边界；
- tool policy：工具权限和拦截规则；
- verifier memory：历史判断标准和反例。

每一类 memory 的更新条件都不一样。

尤其是 safety memory，不能看到一个失败就乱追加规则。它要过 safety-utility gate，否则很容易变成“为了安全啥也不干”。

这也是 SHE 比普通安全 prompt 有意思的地方：它不是把 agent 变胆小，而是学习更细的边界。

## 6. Recuris：memory 不是越多越好，而是要会在长程任务里控制自己

Recuris 是这五篇里最贴近 agent memory 的。

它的核心问题是：长程任务里，history 越来越长，agent 反而更容易迷路。

这听起来很直观。上下文长了以后，agent 可能：

- 忘记当前还有哪些子目标没完成；
- 把旧状态当成新状态；
- 检索到不合适的经验；
- 在错误的时机调用 skill；
- 对工具观察的状态更新不可靠；
- 长程执行中慢慢偏离原计划。

所以 Recuris 不把 memory 简单理解成“更多历史”。它引入一个 Experiential–Working Memory 架构。

在 evolution round $k$，它把 Skill Memory 写成：

$$
M^k=(E^k,W^k,\rho^k,C^k)
$$

其中：

- $E^k$：Experiential Memory，存可复用 skills；
- $W^k$：Working Memory specification，定义当前任务状态 schema 和更新方式；
- $\rho^k$：invocation policy，决定什么时候检索技能、检索哪些技能；
- $C^k$：checker set，检查 observation 是否支持某个状态变化。

这套设计里，working memory 不是聊天记录，而是一个“当前任务状态控制器”。

我觉得这是 Recuris 最关键的点。很多 memory 方法失败，不是因为没存经验，而是因为 agent 不知道当前该用哪条经验。

举个很粗糙的例子：你有一本很全的做菜经验库，但你当前锅里是已经放盐了还是没放盐，你自己搞不清楚，那经验库越全也救不了你。

Recuris 要解决的就是这个问题：

> 用 working memory 维护当前状态，再用当前状态去指导 experiential memory 的 skill invocation。

它的 task-level loop 是：

1. working memory 跟踪当前目标和 verified state；
2. invocation policy 根据当前状态从 experiential memory 里取 skill；
3. agent 执行动作；
4. checker 根据 observation 只提交被证据支持的状态变化。

它的 cross-task evolution loop 是：

1. 收集失败轨迹；
2. 结构化 trace 记录每步的 working state、skill invocation、action、observation；
3. Meta-Agent 把失败定位到某个 memory component；
4. 提出局部 patch；
5. validation gate 比较新旧 memory；
6. 如果修复 source failed task 且不伤 held-out development set，就接受；否则保持不变。

这一步的“保持不变”特别重要。

自进化系统最危险的地方，不是它不能更新，而是它太容易更新。每次失败都改一点，最后可能变成一堆过拟合补丁。

Recuris 的递归是 bounded recursive loop：base LLM、工具、Meta-Agent、localization procedure、patching procedure、validation gate、外层 harness 都固定，只允许 memory-control layer 更新。

这比“让 agent 递归改写一切”安全得多。

实验上，Recuris 在四个长程 benchmark 和十个模型上评估，35/37 个完整 model-benchmark pair 都提升。论文报告在最长任务上收益扩大到 +32.2 points，常见长程失败最多下降 80%。

它还有一个很重要的观察：

> 收益不是来自更多上下文，也不是来自更多 compute，而是来自会自我修订的 memory-control layer。

论文里提到，固定 Skill Memory 本身没有显著收益；把整个 skill library 都塞进 prompt 甚至更贵、更差。这点非常反直觉，但也非常符合我对 agent 的使用体验。

很多时候，agent 不是缺信息，而是缺状态控制。

所以 Recuris 对 memory 方向的启发是：

**memory 的核心不是 storage，而是 control。**

也就是说，我们不应该只问：

> 这条经验存不存？

还应该问：

> 它什么时候触发？触发后改哪个状态？是否有 observation 支撑？是否通过 held-out gate？是否对旧任务有副作用？

这才像一个真正可持续的 memory system。

## 7. 五篇论文放在一起看

这五篇其实是在同一个问题上不断加限制。

MCE 打开了第一扇门：context function 可以自动学。

Meta-Harness 把门开得更大：整个 harness code 可以被搜索。

Harness Updating 马上提醒：更新本身和从更新中获益是两回事。

SHE 把这个问题放进 safety：安全边界不能只靠静态 prompt，要在 trajectory feedback 中局部演化。

Recuris 把它放回 memory：长程 agent 的 memory 不能只是存历史，而要形成带 working state、skill invocation 和 validation gate 的递归系统。

我用一张表总结：

| 论文 | 研究对象 | 关键贡献 | 我觉得最该记住的一句话 |
|---|---|---|---|
| MCE | context function + CE skill | 双层优化，skill 指导 context artifact 生成 | context engineering 可以被 meta-skill 演化 |
| Meta-Harness | harness code | 用完整 filesystem history 和 raw traces 搜索 harness | harness 不是 prompt，而是外部程序 |
| Harness Updating | evaluation decomposition | 拆分 updating 和 benefit | 写了更新，不代表 agent 用得上 |
| SHE | safety harness | system prompt / rule bank / safety memory / tool policy 局部演化 | 安全不是最后拒答，而是 harness 运行时边界 |
| Recuris | memory-control layer | working memory + experiential memory + validation-gated skill memory | memory 不是 storage，而是 control |

它们共同形成一个框架：

$$
\text{Trajectory}
\rightarrow
\text{Diagnosis}
\rightarrow
\text{Localized Update}
\rightarrow
\text{Validation Gate}
\rightarrow
\text{Future Behavior}
$$

如果没有 diagnosis，更新就是瞎改。

如果没有 localized update，memory / harness 会越改越乱。

如果没有 validation gate，自进化就是过拟合机器。

如果 future behavior 没有被拆成 activation、adherence、benefit，那你甚至不知道更新有没有被 agent 真正用上。

## 8. 和上一篇泛读里的论文怎么接

上一篇泛读里我读了很多 test-time compute、verifier、agent reasoning、memory system 的论文。现在回头看，它们可以这样接到这条主线上。

### test-time compute 是原料生产机

Large Language Monkeys、Search-o1、CodeMonkeys 这类论文说明：多采样、多搜索、多 rollouts 可以显著扩大 coverage。

但 coverage 只是说明“某次搜索里可能出现好答案”。它还没有回答：

> 这次好轨迹能不能沉淀成下次不用重新搜索的能力？

所以 test-time search 产生的是经验原料，self-evolving memory / harness 要解决的是经验加工。

### verifier 是选择门

Weak Verifiers、Let’s Verify Step by Step、MATH-SHEPHERD 这类论文提供的是验证思想。

在 MCE 里，它体现为 validation performance；
在 Meta-Harness 里，它体现为 evaluation scores；
在 Harness Updating 里，它体现为 pairwise gain；
在 SHE 里，它体现为 safety-utility validation；
在 Recuris 里，它体现为 held-out development gate。

所以 verifier 不只是给最终答案打分，它也是决定哪些 memory / harness update 可以被写入系统的门。

### ReAct / LATS / SPRINT 提供轨迹结构

agent reasoning/action 论文给出的是轨迹产生机制。

没有轨迹，就没有可诊断的 failure evidence；没有 structured trace，就很难定位是 memory、tool、prompt 还是 skill 出错。

所以这些论文在新主线里不再只是“agent 怎么做任务”，而是：

> agent 怎么产生足够可审计、可归因、可复用的经验轨迹。

### MemGPT / Cartridges / CacheBlend 是 memory 系统前传

MemGPT 强调虚拟上下文管理；Cartridges 把 memory 做成可复用上下文状态；CacheBlend 更偏高效复用 KV/cache。

这些都很重要，但它们更多解决 memory 的存取和效率问题。

而 Recuris、Evo-Memory、MemSkill、MemMA 这一批新论文开始问更进一步的问题：

> memory 能不能从 recall 走向 test-time learning，甚至走向 recursive self-improvement？

这就是我觉得后续值得继续追的地方。

## 9. 我自己的判断

读完这五篇，我现在对 agent memory 有几个判断。

### 判断 1：memory 方向如果只做“更长上下文”，会很快无聊

因为更长上下文当然有用，但它不是最本质的问题。

更本质的是：

- 哪些经验值得存？
- 存成什么形式？
- 什么时候触发？
- 是否和当前状态匹配？
- 是否会干扰旧能力？
- 是否能被弱模型执行？
- 是否能通过安全验证？

这些问题都不是简单扩上下文能解决的。

### 判断 2：自进化 agent 的最小闭环应该带 held-out gate

如果没有 held-out gate，agent 很容易变成 benchmark patch 机器。

尤其是 memory / skill 这种东西，天然容易写成“我上次失败了，所以以后遇到类似字符串就这样做”。短期看很聪明，长期看就是污染 memory。

所以我觉得一个靠谱的 self-evolving memory 系统至少要有：

$$
\text{failed task repair} + \text{held-out no-regression}
$$

只修复当前失败任务不够，还要证明没有伤到一组 anchor tasks。

### 判断 3：安全 memory 应该独立出来

SHE 让我意识到，safety memory 不应该和普通 task memory 混在一起。

普通 memory 追求任务成功率，safety memory 追求边界清晰和风险拦截。两者的优化目标不一样。

如果混在一个 memory bank 里，很容易出现两种坏情况：

1. 安全规则被当成普通建议，优先级不够；
2. 安全规则过度泛化，导致 benign utility 被误杀。

所以 safety memory 应该有自己的 schema、priority、supporting trajectories、benign exemptions 和验证指标。

### 判断 4：弱模型不是不能自进化，而是可能不会用进化结果

Harness Updating 这篇最值得记住的地方就是：小模型也可能写出不错的 harness 更新，但弱 solver 经常 activation / adherence 出问题。

这意味着后续做小模型 agent，不一定只训练它“会写 memory”，更应该训练它：

- 会判断什么时候查 memory；
- 会把 memory 和当前状态对齐；
- 会在长程执行中持续遵守；
- 会在 memory 不适用时拒绝套用。

说白了，就是 memory-following 也是一种能力。

### 判断 5：agent 自进化最危险的不是不会改，而是太会改

这个听起来有点反直觉，但我越读越觉得是这样。

一个完全不会更新的 agent，只是笨。

一个看到任何失败都立刻写 memory、改 prompt、改 tool policy、改 workflow 的 agent，可能很快就会变成一坨过拟合的规则泥潭。

所以安全自进化的关键不是“让 agent 能改自己”，而是“让 agent 只能在足够窄、足够可验证的地方改自己”。

Recuris 的 bounded recursion 和 SHE 的 localized evolution 都是在往这个方向走。

## 10. 如果我要基于这条线做一个 idea

如果顺着这五篇继续想，我觉得一个比较自然的 research idea 是：

> Safety-gated self-evolving memory harness for long-horizon tool agents.

大概可以设计成这样：

### 系统状态

把 agent harness 拆成几块：

$$
H=(P_{\text{sys}},M_{\text{task}},M_{\text{skill}},M_{\text{safe}},Q_{\text{tool}},V)
$$

其中：

- $P_{\text{sys}}$ 是全局行为契约；
- $M_{\text{task}}$ 存任务事实；
- $M_{\text{skill}}$ 存可复用过程；
- $M_{\text{safe}}$ 存安全边界和反例；
- $Q_{\text{tool}}$ 控制工具权限；
- $V$ 是 verifier / gate。

### 执行时

每一步记录：

$$
(w_t, m_t, a_t, o_t, r_t)
$$

其中：

- $w_t$ 是 working state；
- $m_t$ 是被激活的 memory；
- $a_t$ 是 action；
- $o_t$ 是 observation；
- $r_t$ 是 verifier / judge 反馈。

这样失败时可以问：

> 到底是状态没更新，memory 没检索，skill 不适用，tool policy 没挡住，还是模型没遵守？

### 更新时

每个 candidate patch 都必须满足：

$$
\Delta_{\text{source}}>0
$$

并且：

$$
\Delta_{\text{held-out}}\geq 0
$$

如果是 safety patch，还要同时看：

$$
\text{ASR}\downarrow,\quad \text{Benign Utility}\not\downarrow
$$

也就是不能靠“全部拒绝”获得安全。

### 评估时

不要只报 end-to-end success rate，而要拆：

| 指标 | 问题 |
|---|---|
| Update Quality | patch 本身有没有修复失败？ |
| Activation Rate | 该加载的 memory 是否被加载？ |
| State Grounding | memory 是否匹配当前 working state？ |
| Adherence | agent 是否持续遵守 memory / policy？ |
| Held-out Regression | 是否伤到 anchor tasks？ |
| Safety-Utility Tradeoff | 是否靠拒绝牺牲 utility？ |

这个 idea 不一定新，但我觉得它很适合作为后续继续读论文、写博客、甚至做小实验的主轴。

## 11. 结尾：我现在怎么理解 agent memory

读这些论文之前，我对 agent memory 的理解还是偏存储：怎么压缩历史、怎么检索、怎么塞上下文。

读完之后，我现在更愿意把 agent memory 看成：

> 一个会影响未来行为的、可更新的、需要验证门保护的外部控制层。

这句话里每个词都重要。

**会影响未来行为**：如果 memory 只是存着但不改变行动，那它只是日志。

**可更新**：如果 memory 不能从失败中修正，它就不是 self-evolving。

**需要验证门**：如果 memory 可以无约束更新，它迟早会污染自己。

**外部控制层**：它不是模型权重，但它能改变模型在任务中的实际能力边界。

所以 agent memory 的下一步，可能不是做一个更大的记忆库，而是做一个更像操作系统的 memory harness：

- 有工作记忆维护当前状态；
- 有经验记忆提供可复用技能；
- 有安全记忆保护边界；
- 有工具策略限制动作；
- 有验证器决定什么能写入；
- 有 no-regression gate 防止越改越歪。

这条线我觉得是值得继续追的。

尤其是现在 agent 越来越像“会操作电脑的软件系统”，而不是单纯聊天模型。只要 agent 会读文件、写代码、发请求、调用工具，它的安全问题和能力增长问题就一定会落到 harness 层。

最后的关键问题也就变成：

> 我们能不能让 agent 学会更新自己，但只在它真的知道自己在改什么的时候更新？

这大概就是我目前对 **agent memory × 安全自进化** 这个方向最核心的理解。


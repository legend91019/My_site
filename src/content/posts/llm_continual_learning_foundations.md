---
title: "语言模型如何持续学习：从灾难性遗忘到重放、正交子空间与 LoRA"
summary: "从持续学习的基本公式出发，读懂 LAMOL、LFPT5、Progressive Prompts 和 O-LoRA 的核心思路与限制。"
date: 2026-07-11
tags: ["大模型","持续学习","灾难性遗忘"]
category: "学习指南"
draft: false
---

如果一个语言模型今天学会了情感分类，明天又去学 SQL 生成，后天再去学摘要，它会发生什么？

最朴素的答案是：继续微调就好了。

但持续学习里最经典、也最讨厌的问题恰恰在这里：模型学新任务的时候，经常会把旧任务忘掉。这个现象叫 **catastrophic forgetting**，中文一般翻译成灾难性遗忘。

这篇我想把这件事从头捋一遍。读者只需要知道 Transformer 是什么，不需要提前懂 LoRA，也不需要提前懂 continual learning。我的目标不是把所有论文排成一个排行榜，而是回答一个更基础的问题：

> 语言模型如果要持续学习，大家到底在保护什么、牺牲什么、假设什么？

我会先讲形式化定义和指标，再讲几类经典方法，最后重点读四条语言模型相关路线：

+ **LAMOL**：让同一个 LM 同时当解题器和伪样本生成器
+ **LFPT5**：冻结 T5，用 prompt tuning 做 few-shot 持续学习
+ **Progressive Prompts**：每个任务新增 soft prompt，并把历史 prompt 串起来
+ **O-LoRA**：每个任务新增 LoRA，并约束 LoRA 子空间彼此正交

这里要先打一个预防针：Progressive Prompts 的源协议里，每个任务采样 16 个 examples；这些论文的设置并不完全一样。数据集、backbone、是否 few-shot、是否需要 task ID、是否允许 replay 都不同。所以后面所有数字都只能在各自实验协议里理解，不能粗暴地说“某方法一定强于某方法”。

---

## 一、持续学习到底在学什么？

普通微调通常默认我手里有一个固定训练集：

$$
D = \{(x_i, y_i)\}_{i=1}^{n}
$$

持续学习不一样。它面对的是一个任务流：

$$
\mathcal{T} = (T_1, T_2, \dots, T_T)
$$

第 $t$ 个任务有自己的数据：

$$
D_t = \{(x_i^t, y_i^t)\}_{i=1}^{n_t}
$$

模型在学习 $T_t$ 的时候，通常只能访问当前任务数据 $D_t$，不能随便回到过去把 $D_1,\dots,D_{t-1}$ 全部再训一遍。学完第 $t$ 个任务之后，我们又会在所有已见任务上评估：

$$
T_1,\dots,T_t
$$

这就带来一个很直接的矛盾：

+ 如果模型参数很容易被新任务改变，它会有很强的 **plasticity**，也就是可塑性，能快速适应新任务。
+ 如果模型参数不容易被改变，它会有很强的 **stability**，也就是稳定性，能保留旧任务。

持续学习基本就在这两者之间拉扯：

**太稳定，学不进新东西；太可塑，旧东西会掉。**

这就是 stability-plasticity dilemma。

语言模型让这个问题更微妙。传统 CL 论文里，任务经常是图像分类：今天认猫狗，明天认车船。可语言模型里，一个“任务”可能是问答、分类、摘要、语义解析、对话状态追踪，也可能只是同一个任务类型下换了领域。它不只是换标签空间，还可能换输入输出格式。

所以，在读语言模型持续学习论文时，我现在会先问三个问题：

+ 学到新任务时，旧任务数据还能不能被看见？
+ 推理时，模型知不知道当前样本属于哪个任务？
+ 模型容量会不会随任务数增长？

这三个问题往往比单个准确率数字更重要。

---

## 二、task-incremental、domain-incremental、class-incremental，不要混在一起

持续学习里经常会看到三个词：

| 设置 | 核心变化 | 推理时常见假设 | 例子 |
|---|---|---|---|
| Task-incremental | 任务类型变了 | 通常知道 task ID | 先学情感分类，再学问答 |
| Domain-incremental | 任务类型不变，数据分布变了 | 不一定知道 domain ID | 都做分类，但从新闻换到商品评论 |
| Class-incremental | 标签类别逐步增加 | 通常不知道任务边界 | 今天学猫狗，明天加汽车飞机 |

语言模型论文里这三个边界会变得有点黏。比如 LFPT5 会明确区分同一 task type 下的新 domain，以及不同 task type 之间的迁移；Progressive Prompts 的实验虽然多是文本分类，但方法定义里假设训练和推理都知道 task identity；O-LoRA 则强调推理时不依赖 task ID，更接近开放式 instruction tuning 的需求。

我觉得这里最容易踩坑的是：**task-ID-free 的方法不一定更准，task-ID-aware 的方法也不一定不合理。**

如果你的产品就是一个明确路由系统，比如“这个请求就是摘要任务”，task ID 是自然存在的。可如果你希望一个聊天模型面对任意指令都能自己判断，那 task-ID-free 就更接近真实使用方式。

---

## 三、怎么量化遗忘？

设 $R_{i,j}$ 表示：

> 模型学完第 $i$ 个任务后，在第 $j$ 个任务测试集上的分数。

这样就可以得到一个三角形评估矩阵。比如 $R_{5,2}$ 就是“学完第 5 个任务后，对第 2 个任务的表现”。

最常见的最终平均准确率是：

$$
AA_T = \frac{1}{T}\sum_{j=1}^{T}R_{T,j}
$$

它回答的是：所有任务都学完之后，平均还剩多少能力。

遗忘量可以写成：

$$
F_T = \frac{1}{T-1}\sum_{j=1}^{T-1}
\left(\max_{l \in \{1,\dots,T-1\}} R_{l,j} - R_{T,j}\right)
$$

它比较的是：旧任务曾经达到过的最好表现，和最终表现之间差多少。

Backward Transfer 看新任务学习对旧任务的影响：

$$
BWT = \frac{1}{T(T-1)/2}\sum_{i=2}^{T}\sum_{j=1}^{i-1}(R_{i,j}-R_{j,j})
$$

如果 $BWT < 0$，通常说明学新任务伤害了旧任务；如果 $BWT > 0$，说明后来的学习反而帮助了旧任务。

Forward Transfer 看旧任务学习对未来任务的帮助：

$$
FWT = \frac{1}{T(T-1)/2}\sum_{i=1}^{T-1}\sum_{j=i+1}^{T}(R_{i,j}-b_j)
$$

这里 $b_j$ 是在学习任务 $j$ 之前的参考基线，可以是随机、零样本或某种独立初始化基线。

除了这些分数，我还会额外看三类“成本指标”：

| 指标 | 要问的问题 |
|---|---|
| Replay | 是否保存旧样本，或生成伪旧样本 |
| Parameter growth | 参数是否随任务数线性增长 |
| Task-ID assumption | 训练和推理时是否需要知道任务身份 |

很多论文的核心差异，不在公式第一眼看起来多漂亮，而在这些成本藏在哪里。

---

## 四、持续学习方法的大地图

在语言模型之前，持续学习已经有一套很经典的分类法。

| 路线 | 直觉 | 典型方法 | 主要代价 |
|---|---|---|---|
| Replay | 学新任务时复习旧任务 | ER、生成式 replay、LAMOL | 要存样本，或要有可靠生成器 |
| Regularization | 重要参数别乱动 | EWC、MAS、LwF | 保护过强会学不动新任务 |
| Gradient / subspace | 新梯度别干扰旧梯度 | GEM、A-GEM、OGD、O-LoRA | 需要估计旧任务方向或子空间 |
| Parameter isolation | 不同任务用不同参数 | Progressive Networks、PackNet、Prompts | 容量随任务增长，常需要 task ID |
| PEFT | 只调小模块 | Prompt、Adapter、LoRA | 参数少，但路由和表达能力有新问题 |

语言模型时代，PEFT 变得特别重要。原因很现实：如果 backbone 是几十亿参数，你很难为每个任务都复制一份模型，也不想每来一个任务就全量微调。

所以后面几篇论文其实都在围绕一个共同问题打转：

> 能不能只动很小一部分参数，同时减少遗忘？

---

## 五、四个基础方法：EWC、LwF、GEM、OGD

先看 EWC。它的直觉是：旧任务里重要的参数不要动太多。

设 $\theta_i^*$ 是旧任务学完后的参数，$F_i$ 是第 $i$ 个参数的重要性估计，常用 Fisher 信息的对角近似。学习新任务时，损失变成：

$$
L(\theta)=L_t(\theta)+\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta_i^*)^2
$$

这项惩罚的意思很朴素：如果某个参数对旧任务很重要，偏离旧值就要付出更大代价。

LwF 走的是蒸馏路线。它不一定保存旧数据，而是让新模型在一些输入上保留旧模型的输出分布：

$$
L=L_{\text{new}}+\lambda L_{\text{distill}}
$$

其中：

$$
L_{\text{distill}}
=-\sum_c p_T^{\text{old}}(c)\log p_T^{\text{new}}(c)
$$

温度 softmax 写成：

$$
p_T(c)=\frac{\exp(z_c/T)}{\sum_{c'}\exp(z_{c'}/T)}
$$

它保护的不是某个参数，而是模型的行为。

GEM 更直接地看梯度。设当前任务梯度是 $g$，旧任务记忆样本上的梯度是 $g_k$。GEM 希望当前更新不要让旧任务损失上升：

$$
\min_\theta \ell(f_\theta(x,t),y)
\quad
\text{s.t.}\quad
\langle g,g_k\rangle \ge 0,\ \forall k<t
$$

如果当前梯度会伤害旧任务，就把它投影到满足约束的方向。

OGD 把这个想法变成正交投影。设 $U$ 张成旧梯度子空间，当前梯度是 $g$，则：

$$
\tilde g = g-U(U^\top U)^{-1}U^\top g
$$

然后用：

$$
\theta \leftarrow \theta-\eta \tilde g
$$

也就是说，新的更新尽量走到旧梯度子空间的正交补里。

这四个方法很适合当“坐标系”：EWC/LwF 保护参数或输出，GEM/OGD 保护梯度方向。后面 O-LoRA 的想法，其实就很像在 LoRA 参数里做一种低秩子空间版本的 OGD。

---

## 六、LAMOL：让语言模型自己生成旧题来复习

LAMOL 的问题意识很清楚：之前很多 lifelong learning 工作集中在图像或同类 NLP 任务上，但语言任务可能彼此很不一样。如果每学一个任务都保留真实旧样本，会有存储和隐私问题；如果额外训练一个生成器，系统又变复杂。

LAMOL 的直觉是：

**既然 language model 本来就会生成文本，那能不能让同一个模型既做任务求解器，又做旧样本生成器？**

它把所有任务都转成 SQuAD-like 的格式：

```text
context + question + answer
```

并加入三个特殊 token：

+ `ANS`：question 和 answer 的分隔符
+ `EOS`：样本结束符
+ `GEN`：生成伪样本的起始符

对第 $i$ 个任务 $T_i$，在训练前先生成旧任务伪样本 $T_i'$，数量为：

$$
\left|T_i'\right|=\gamma \left|T_i\right|
$$

其中 $\gamma$ 是 sampling ratio。

训练时，每个样本同时构造成 QA format 和 LM format，联合损失是：

$$
L=L_{QA}+\lambda L_{LM}
$$

$L_{QA}$ 让模型读 context/question 后生成 answer；$L_{LM}$ 让模型从 `GEN` 开始生成完整样本。论文默认超参里 top-k sampling 的 $k=20$，$\lambda=0.25$。

如果使用 task-specific generation tokens，那么训练第 $i$ 个任务时，会给前 $i-1$ 个任务各生成：

$$
\frac{\gamma}{i-1}|T_i|
$$

条伪样本。这样是为了避免早期任务在伪样本分布里被指数式稀释。

训练流程可以概括成：

1. 把当前任务数据转成统一 QA 格式。
2. 如果不是第一个任务，用当前模型通过 top-k sampling 生成旧任务伪样本。
3. 丢弃不含恰好一个 `ANS` 的生成样本；论文报告这种情况约 0.5%-1%。
4. 混合伪旧样本和当前任务样本。
5. 同一步里优化 $L_{QA}$ 和 $L_{LM}$。

LAMOL 的主实验用最小版 pre-trained GPT-2；抽取文本里没有给出确切参数量。每个任务训练 9 epochs，推理用 greedy decoding。

它的五个 decaNLP 任务包括：

| 任务 | 数据集 | 训练 / 测试 | 指标 |
|---|---|---:|---|
| Question answering | SQuAD | 87,599 / 10,570 | nF1 |
| Semantic parsing | WikiSQL | 56,355 / 15,878 | lfEM |
| Sentiment analysis | SST | 6,920 / 1,821 | EM |
| Semantic role labeling | QA-SRL | 6,414 / 2,201 | nF1 |
| Goal-oriented dialogue | WOZ | 2,536 / 1,646 | dsEM |

baselines 包括 Fine-tune、Multitask learning、Online EWC、MAS、GEM；文本分类实验还比较 MBPA++。

几个关键数字：

| 设置 | LAMOL 结果 | 参照 |
|---|---:|---|
| SST / QA-SRL / WOZ 三任务置换 | `LAMOL^{0.2}_{GEN}` 79.7，`LAMOL^{0.2}_{TASK}` 79.5 | Multitasked 81.5；Fine-tuned 39.3；EWC 47.0；MAS 41.6；GEM 42.8 |
| 五个 decaNLP 顺序任务 | `LAMOL^{0.2}_{TASK}` 74.3，`LAMOL^{0.2}_{REAL}` 76.0 | Multitasked 76.6 |
| 文本分类四种任务顺序 | `LAMOL^{0.2}_{TASK}` 平均 76.5 | MBPA++ 原论文 70.7，复现版 74.2 |

这里最值得注意的是：LAMOL 并不保存真实旧样本，主要靠生成式 pseudo replay。基础版参数不随任务增长；如果加 task-specific tokens，只会轻微增大 vocab 和 embedding。

但它也有明显限制。

第一，pseudo-data 没有真实旧样本高效。论文里 5% real samples 已经可以超过 20% pseudo-samples。

第二，如果不用 task-specific tokens，早期任务分布会被后续任务稀释。

第三，即便用了 task-specific tokens，生成也会出现“chaos”：token 指示的是一个任务，生成内容却像另一个任务。

所以我对 LAMOL 的理解是：它证明了“LM 自己生成旧题复习”这条路可行，但这个生成器的质量，会直接变成持续学习的上限。

---

## 七、LFPT5：把 LAMOL 的生成复习搬到 frozen T5 + prompt tuning

LFPT5 研究的是 Lifelong Few-shot Language Learning。它比 LAMOL 更进一步：每个新任务或新 domain 只有少量样本，而且 backbone T5 冻结，只学习 prompt embeddings。

如果读者还不熟 LoRA/PEFT，可以先把 prompt tuning 理解成：

> 不改 Transformer 主体参数，只在输入前面接一串可训练的“软 token”。

LFPT5 的基本分解写成：

$$
\arg\max_\phi \log p(\phi|D_{task},D_{pre})
\approx
\arg\max_\phi
[\log p(\phi|D_{task},\theta)+\log p(\theta|D_{pre})]
$$

这里 $\theta$ 是预训练 T5，$\phi$ 是 prompt embeddings。因为 $\theta$ 冻结，所以适配主要发生在 $\phi$ 上。

任务求解损失是：

$$
L_{\phi}^{task}
=-\sum_{i=1}^{n}\log p(Y_i|[P,X_i],\phi,\theta)
$$

$P$ 是 prompt tokens，$X_i$ 是输入文本，$Y_i$ 是输出文本。

生成损失是：

$$
L_{\phi}^{lm}
=-\sum_{i=1}^{n}\log p([X_i,Y_i]|[G,P],\phi,\theta)
$$

$G$ 是 task/domain-specific generation token。

LFPT5 还加入 KL 一致性损失，让新 prompt 在伪样本上的输出分布接近旧 prompt：

$$
L_{\phi}^{KL}
=
\sum_{i=1}^{m}\sum_{j=1}^{t}
D_{KL}
\left(
p_j(V|[P,\tilde X_i],\phi_0,\theta)
\;\|\;
p_j(V|[P,\tilde X_i],\phi,\theta)
\right)
$$

其中 $\phi_0$ 是上一阶段 prompt，$\tilde X_i$ 是伪样本输入，$V$ 是 T5 vocabulary。

总损失是：

$$
L_\phi
=L_\phi^{task}
+\lambda_{lm}L_\phi^{lm}
+\lambda_{kl}L_\phi^{KL}
$$

训练流程是：

1. 把 NER、分类、摘要都统一成 text-to-text。
2. 同一 task type 内遇到新 domain 时，先用 `GEN_*` token 生成旧 domain 伪标注样本。
3. 将伪样本和当前 few-shot 数据混合训练。
4. 用 KL loss 保持旧 prompt 与新 prompt 的输出一致性。
5. 遇到新 task type 时，新增一套 prompt token，只训练新 prompt，旧 prompt 冻结。

它的主干是 T5-Large，附录还做了 T5-Base。每个 task-type prompt 是 300 个 tunable tokens；论文称新增一个 task type 约增加预训练 T5 参数量的 0.04%。抽取文本里没有给出 T5-Large/T5-Base 的绝对参数量。

数据集和指标如下：

| 任务类型 | 数据集 | few-shot 设置 | 指标 |
|---|---|---|---|
| NER | CoNLL03、OntoNotes | 每类 16 个样本 | F1 |
| 分类 | AGNews、Amazon、DBPedia、Yahoo | 每类 16 个样本 | Accuracy |
| 摘要 | CNNDM、WikiHow、XSum | 每 domain train/valid 各 64 条 | A-RG |

A-RG 是 ROUGE-1、ROUGE-2、ROUGE-L 的平均值。

baselines 包括 FT、PT、EWC-PT、MAS-PT、EWC-FT、MAS-FT、PT-R、MT-PT；跨 task type 时还比较 MT-FT 和 AdapterFusion。

关键结果要分任务看：

| 设置 | LFPT5 | 更强参照 | 更弱参照 |
|---|---:|---:|---:|
| NER 最终 F1 | 47.59±2.16 | PT-R 48.72±0.9；MT-PT 54.32±0.88 | PT 44.34±0.46；FT 43.07±1.48 |
| 分类最终 accuracy | 52.71±4.19 | PT-R 67.23±1.36；MT-PT 76.08±0.77 | FT 40.11±7.76；EWC-FT 40.60±3.02 |
| 摘要最终 A-RG | 17.05±0.92 | PT-R 17.48±0.25；MT-PT 19.78±0.70 | PT 15.67±0.24；FT 15.71±1.35 |

跨不同 task type 时，LFPT5 整体优于 MT-FT、MT-PT、AdapterFusion。比如 `Summ-Class-NER` 顺序下，`LFPT5 w.o. FKT` 得到 `25.48, 84.75, 63.28`，`LFPT5 with FKT` 得到 `25.48, 86.00, 62.44`，而 MT-PT 是 `24.16, 85.50, 50.80`。

LFPT5 的 replay 是生成式 pseudo labeled replay，不保存真实旧样本。参数增长取决于任务类型：同一 task type 的新 domain 不扩 backbone；新 task type 会新增一套 300-token prompt。

task-ID 假设要小心：生成旧 domain 样本时需要 domain-specific generation token；跨 task type 推理时需要选择对应 task prompt，所以至少默认知道 task type。

它的限制也很清楚。

第一，伪样本质量仍然是瓶颈。NER、分类、摘要都明显低于 PT-R 和 MT-PT。

第二，摘要任务里的伪摘要容易模糊，作者甚至采用生成文档的 lead-3 句子作为 pseudo summary。

第三，prompt 可调参数少，few-shot 本身又难，所以 PT-based EWC/MAS 在一些旧任务保持上几乎失效。

所以 LFPT5 给我的启发是：当 backbone 冻结之后，持续学习问题没有消失，只是被压缩进 prompt 的小空间里。这个小空间更省参数，但也更容易容量不够。

---

## 八、Progressive Prompts：不改旧 prompt，就不会忘旧 prompt

Progressive Prompts 的出发点和 LFPT5 不太一样。它不想依赖 replay，也不想持续改同一个 prompt。它的做法更像 progressive networks 的轻量版：

> 每来一个新任务，就学一个新 soft prompt；旧 prompt 冻结；输入时把当前 prompt 和历史 prompt 串起来。

单任务 prompt tuning 目标可以写成：

$$
\max_{\theta_P}
\sum_{x,y\in T}\log p_{\theta,\theta_P}(y|[P;x])
$$

其中 $\theta$ 是冻结的预训练 LM，$\theta_P$ 是 prompt 参数。

第 $k$ 个任务的 continual 目标是：

$$
L(\theta_{P_k})
=
-\sum_{x,y\in T_k}
\log p(y|[P_k,\dots,P_1,x],
\theta,\theta_{P_1},\dots,\theta_{P_k})
$$

训练第 $k$ 个任务时，只更新 $P_k$；$P_1,\dots,P_{k-1}$ 和 base model 都冻结。

它还用了 residual MLP 做 prompt 重参数化：

$$
P'_k=MLP(P_k)+P_k
$$

直觉是让 prompt tuning 更稳定。训练完成后，MLP 重参数化模块可以丢弃，只保留投影后的 prompt。

训练流程非常清楚：

1. 新任务到来，创建新 prompt $P_k$。
2. 输入前拼接 $[P_k,\dots,P_1,x]$。
3. 只训练当前 prompt。
4. 学完后冻结当前 prompt，留给后续任务复用。

主实验用 T5-Large 和 BERT-base。prompt 长度上，BERT 单任务 20 tokens；T5 标准 few-shot benchmark 用 50 tokens，长序列实验用 10 tokens。论文摘要称学习参数小于 0.1%；抽取文本没有给出模型绝对参数量。

数据设置有两类：

| 设置 | 数据集 | 规模 | 指标 |
|---|---|---|---|
| 标准 CL benchmark | AG News、Amazon、Yelp、DBpedia、Yahoo Answers | BERT：每任务 115,000 train / 7,600 test；T5：沿用 LFPT5 few-shot 协议 | 最终 averaged accuracy |
| 15 任务长序列 | 5 个分类集 + GLUE/SuperGLUE/IMDB 等 | 每类 20 / 200 / 1000 训练样本 | 最终 averaged accuracy，附录报 FWT/BWT |

baselines 包括 Finetune、EWC、A-GEM、Experience Replay、MBPA++、IDBR、Per-task prompts、PromptTuning、LFPT5；表中还给出 Per-task Finetune 和 MTL 作参照。

关键数字如下：

| 设置 | Progressive Prompts | 参照 |
|---|---:|---|
| T5 few-shot benchmark | 平均 75.1 | LFPT5 52.7；Finetune 28.5；Replay 38.0；EWC 40.6；Per-task Finetune 70.0 |
| BERT benchmark | 平均 77.9 | IDBR 76.3；MBPA++ 70.6；A-GEM 66.9 |
| 15 任务 T5-Large | 20/200/1000 每类样本：76.2 / 78.7 / 79.5 | LFPT5：54.3 / 58.2 / 69.2 |
| 15 任务 BERT-base | 53.5 / 66.9 / 69.3 | IDBR：36.8 / 47.9 / 52.2 |

这个方法的优点很漂亮：无 data replay，旧 prompt 冻结，所以旧 prompt 自己不会被改坏；历史 prompt 又能给新任务提供 forward transfer。

但代价也同样漂亮地摆在那里。

第一，参数随任务数线性增长。虽然每个 prompt 很小，但任务越多，总 prompt 越长。

第二，论文在 continual learning 定义里明确假设训练和推理都能访问 task identity。也就是说，推理时你知道该用哪个任务设定。

第三，它主要服务于已见任务的持续学习，而不是 O-LoRA 那种强调 unseen-task generalization 的设置。

所以 Progressive Prompts 更像一个非常干净的上界式方案：如果我愿意给每个任务一段独立 prompt，并且推理时知道任务身份，那我可以用很少参数换来很强的抗遗忘。

---

## 九、先补一小段：LoRA 到底是什么？

读 O-LoRA 前，需要先知道 LoRA 的最小概念。

Transformer 里有很多线性层，权重矩阵可以写成：

$$
W_{init}\in\mathbb{R}^{d\times k}
$$

普通全量微调会直接更新 $W_{init}$。LoRA 的想法是：不直接改整个矩阵，而是用一个低秩更新表示变化：

$$
W_{init}+\Delta W
=W_{init}+AB
$$

其中：

$$
A\in\mathbb{R}^{d\times r},\quad
B\in\mathbb{R}^{r\times k},\quad
r\ll \min(d,k)
$$

前向传播里，如果原来是：

$$
h=W_{init}x
$$

LoRA 后就是：

$$
h=W_{init}x+\Delta Wx
=W_{init}x+ABx
$$

这就是参数高效的原因：只训练小矩阵 $A$ 和 $B$，不训练完整的 $W_{init}$。

但这也埋下一个问题：如果每个任务都学一组 LoRA，这些低秩更新之间会不会互相干扰？

O-LoRA 就是沿着这个问题走的。

---

## 十、O-LoRA：把不同任务放进彼此正交的 LoRA 子空间

O-LoRA 的完整标题是 *Orthogonal Subspace Learning for Language Model Continual Learning*。它的目标比 Progressive Prompts 更接近开放式语言模型使用：

+ 不保存旧数据，rehearsal-free
+ 参数高效
+ 推理时不依赖 task ID
+ 尽量保留 unseen tasks 泛化

它的核心直觉是：

**LoRA 的低秩子空间可以近似表示一个任务的重要更新方向；如果新任务的 LoRA 子空间与旧任务 LoRA 子空间正交，就能减少干扰。**

对第 $t$ 个任务，LoRA 参数写成：

$$
A_t=[a_1^t,a_2^t,\dots,a_r^t]
$$

它张成的子空间是：

$$
U_t=span\{a_1^t,a_2^t,\dots,a_r^t\}
$$

$B_t=[b_1^t,\dots,b_r^t]$ 则可以理解成这些列向量的线性权重系数。

两个子空间正交的条件是：

$$
\langle u,w\rangle=0,\quad \forall u\in U,\ w\in W
$$

对 LoRA 的 $A$ 矩阵，O-LoRA 用：

$$
O_{i,t}=A_i^\top A_t=0
$$

表示第 $i$ 个旧任务和第 $t$ 个新任务的 LoRA 子空间正交。

训练目标写成：

$$
\sum_{x,y\in D_t}\log p_{\Theta}(y|x)
+
\lambda_1\sum_{i=1}^{t-1}L_{orth}(A_i,A_t)
$$

其中：

$$
L_{orth}(A_i,A_t)
=
\sum_{j,k}\|O_{i,t}[j,k]\|^2
$$

也就是说，如果 $A_i^\top A_t$ 里还有非零相关性，就惩罚它。

训练流程是：

1. 用统一 instruction schema 表示任务：`Task Definition + Options + Text + Answer`。
2. 第 $t$ 个任务到来时，新增一组 LoRA 参数 $\{A_t,B_t\}$。
3. 固定旧任务 LoRA 参数 $\{A_i,B_i|i<t\}$。
4. 训练当前任务，同时加入正交正则。
5. LoRA 只作用在 attention 的 $W_q$ 和 $W_v$。
6. 历史 LoRA 可以 merge 回初始参数，缓解训练时显存膨胀。

merge 写成：

$$
W_{init}:=W_{init}+\sum_{i=1}^{t}A_iB_i
$$

O-LoRA 主结果用 T5-large；unseen-task 实验用 LLaMA-7B；分析里还比较 T5-base 和 T5-xl。抽取文本里只有 LLaMA-7B 的绝对参数量明确写出 7B，T5 各型号没有打印绝对参数量。

数据和指标：

| 设置 | 数据 | 指标 |
|---|---|---|
| 标准 CL benchmark | AG News、Amazon reviews、Yelp reviews、DBpedia、Yahoo Answers，三种顺序 | AA |
| 15 任务长序列 | 与 Progressive Prompts 一致；主文每任务 1000 训练样本，每类 500 验证 | AA |
| unseen-task generalization | 先 Alpaca instruction tuning，再顺序学 CL benchmark，用 MMLU zero-shot 测泛化 | MMLU zero-shot + CL benchmark |

baselines 包括 SeqFT、SeqLoRA、IncLoRA、Replay、EWC、LwF、L2P、LFPT5、ProgPrompt、PerTaskFT、MTL。

关键数字：

| 设置 | O-LoRA | 参照 |
|---|---:|---|
| 标准 CL benchmark，T5-large | 三个顺序 75.4 / 75.7 / 76.3，平均 75.8 | LFPT5 平均 72.7；ProgPrompt 75.1；MTL 80.0 |
| 15 任务长序列，T5-large | 三个顺序 72.3 / 64.8 / 71.6，平均 69.6 | LFPT5 69.2；ProgPrompt 77.9；PerTaskFT 78.1；MTL 76.5 |
| unseen-task，LLaMA-7B | `Alpaca-OLoRA-CL` MMLU zero-shot 33.6，CL order 1 为 76.8 | `Alpaca-LoRA-CL` 23.3 / 46.7；`Alpaca-inc-LoRA-CL` 28.6 / 33.1 |

O-LoRA 的假设和代价很值得单独看。

+ replay：不需要历史数据回放。
+ 参数增长：每个任务新增一组 LoRA，但可以 merge 缓解显存问题。
+ task ID：推理时不依赖 task ID；但论文限制里也说，训练时仍需要 task identification 来为每个任务训练不同 LoRA。
+ unseen tasks：论文用 Alpaca + MMLU zero-shot 支撑“更保泛化”的说法。

它的限制也不能忽略。

第一，在 15 任务长序列上，O-LoRA 明显低于 ProgPrompt、PerTaskFT、MTL。这说明“无 task-ID、保泛化”的设置是有代价的。

第二，参数仍随任务数增长，只是比整模微调轻得多。

第三，也是我觉得最关键的一点：O-LoRA 的核心论证是“LoRA 子空间可以近似历史梯度子空间”。这很有启发性，也有实验支持，但它不是严格理论保证。

---

## 十一、把四篇放到一张表里看

如果只看方法名，很容易觉得它们都在做“防遗忘”。但它们保护的东西不一样。

| 方法 | 保护方式 | 是否 replay | 参数增长 | 推理 task ID | 主要优势 | 主要代价 |
|---|---|---|---|---|---|---|
| LAMOL | LM 生成旧任务伪样本 | 生成式 replay | 基础版不增长；task token 略增 | 求解时未显式要求 | 单模型兼任 solver/generator | 伪样本质量和任务混乱 |
| LFPT5 | frozen T5 + prompt + 伪样本 + KL | 生成式 replay | 新 task type 增 300-token prompt | 至少需要 task type | few-shot、参数省 | 小 prompt 容量和伪样本质量限制 |
| Progressive Prompts | 每任务新增并冻结 prompt | 无 data replay | 每任务线性增长 | 需要 | 抗遗忘强，forward transfer 清楚 | prompt 越来越长，依赖 task ID |
| O-LoRA | 每任务 LoRA 子空间正交 | 无 data replay | 每任务新增 LoRA，可 merge | 推理不需要，训练需要 | 更接近 instruction tuning 与 unseen-task | 长序列上弱于 task-ID-aware 方法 |

如果从“更像真实聊天模型部署”这个角度看，O-LoRA 的假设更诱人，因为推理时不需要任务 ID。但如果从“已知任务集合里的最终平均准确率”看，Progressive Prompts 这种显式任务参数隔离非常强。

这就是为什么我不太愿意把它们排成单一名次。

它们解决的是同一个大问题，但坐标轴不同。

---

## 十二、我现在如何理解这条发展线？

LAMOL 的关键是：

> 语言模型可以自己生成旧样本来复习。

LFPT5 的关键是：

> 这件事可以搬到 frozen backbone + prompt tuning 的 few-shot 场景里。

Progressive Prompts 的关键是：

> 如果不改旧 prompt，遗忘可以被结构性地压低；但你要接受任务参数和 task ID。

O-LoRA 的关键是：

> 也许我们可以不靠 task-ID routing，而是在低秩更新空间里把任务方向分开。

这条线一路走下来，会发现语言模型持续学习的核心矛盾并没有消失，只是在不同位置重新出现：

+ replay 方法把难点放在“伪样本是否可靠”
+ prompt 方法把难点放在“prompt 容量和任务路由”
+ LoRA 方法把难点放在“低秩子空间是否真的隔离了知识”
+ 正交方法把难点放在“正交约束是否既防遗忘又不伤迁移”

我觉得 O-LoRA 最值得继续追问的，就是下面几个假设。

第一，$A_t$ 张成的 LoRA 子空间，真的足以代表任务的关键更新方向吗？LoRA 更新是 $AB$，只约束 $A$ 的正交，是否足以约束整个 $\Delta W$？

第二，任务之间真的应该严格正交吗？有些任务共享语义、格式、推理能力。过度正交可能防止干扰，也可能阻断正迁移。

第三，merge 之后的参数空间还保持我们想象中的隔离吗？训练时的 LoRA 子空间是分开的，但合并回 $W_{init}$ 后，不同任务更新已经叠加到了同一个权重矩阵里。

第四，优化器会不会绕过正交约束？尤其 LoRA 是双线性参数化 $AB$，AdamW 又有自适应动量和权重衰减。即便某个显式子空间正交，真实更新轨迹也未必那么听话。

这些问题正好通向下一篇：**正交 LoRA 真能防止遗忘吗：从子空间复用到双线性优化失稳**。

下一篇我会先去看 Bilinear Optimization Divergence（BOD），再接着看 WRP 和 Scaling Buffer Effect。到那时，问题会从“持续学习有哪些基本方法”变成更尖锐的：

> 正交到底是在保护知识，还是在消耗模型未来可学习的空间？

---

## 参考

+ LAMOL: Language Modeling for Lifelong Language Learning, arXiv:1909.03329, https://arxiv.org/abs/1909.03329
+ LFPT5: A Unified Framework for Lifelong Few-shot Language Learning Based on Prompt Tuning of T5, arXiv:2110.07298, https://arxiv.org/abs/2110.07298
+ Progressive Prompts: Continual Learning for Language Models, arXiv:2301.12314, https://arxiv.org/abs/2301.12314
+ Orthogonal Subspace Learning for Language Model Continual Learning, ACL 2023 Anthology / O-LoRA, https://aclanthology.org/2023.findings-emnlp.715/
+ LoRA: Low-Rank Adaptation of Large Language Models, arXiv:2106.09685, https://arxiv.org/abs/2106.09685

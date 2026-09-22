---
title: "持续学习综述：从视觉经典方法到语言模型的实验协议"
summary: "梳理持续学习的定义、任务形态、遗忘与迁移指标，以及 replay、正则化、梯度投影、参数隔离和 PEFT 的方法脉络；对照 O-LoRA 等原论文厘清语言模型常用的短序列、长序列与生成任务协议。"
date: 2026-09-06
tags: ["大模型", "持续学习", "灾难性遗忘", "综述"]
category: "论文解读"
draft: false
---

刚开始读持续学习论文时，很容易遇到一个问题：明明研究的是语言模型，却发现很多基础概念、算法和理论来自视觉论文。EWC、iCaRL、GEM、GPM、L2P、InfLoRA 经常一起出现在参考文献里，但它们并不都在研究同一种任务。

这不是阅读方向偏了。**持续学习是一组关于顺序学习、知识保持和资源限制的问题，视觉和语言只是不同的落地场景。** 视觉研究积累了大量经典方法和规范化实验；语言模型继承了这些思路，也带来了预训练知识、自然语言指令、生成输出和通用能力保持等新问题。

本文是一篇面向入门读者的小综述。前半部分建立领域地图，讲清楚每条路线的动机、代表工作和代价；后半部分具体解释语言模型论文反复使用的实验协议，尤其是 LFPT5、Progressive Prompts、O-LoRA 之后沿用的短任务流与长任务流。目标是读完之后，再看一篇持续学习论文，能判断它在解决哪个问题、比较条件是什么，以及结果可以说明到哪一步。

本文是系列第一篇。[第二篇](/posts/orthogonal_lora)进一步展开 PEFT 与正交 LoRA 的 methodology；[第三篇](/posts/on_the_instability_of_orthogonal_lora)介绍我关于正交 LoRA 失稳的论文。

## 一、持续学习到底在学什么？

### 1. 从一次训练到一条任务流

普通监督学习通常假设训练数据来自相对固定的分布，可以反复打乱并训练。持续学习则让数据按时间到达：

$$
D_1,D_2,\ldots,D_T.
$$

完成任务 $t$ 后，模型由 $\theta_{t-1}$ 变成 $\theta_t$。训练时可能只能访问 $D_t$，但最终希望模型在所有已见任务上都表现良好。一个概念上的目标是：

$$
\min_{\theta_t}\sum_{i=1}^{t}w_i
\mathbb E_{(x,y)\sim P_i}\bigl[\ell(f_{\theta_t}(x),y)\bigr].
$$

困难在于：目标涉及历史分布 $P_i$，训练时却未必还能访问它们。不同方法的差别，往往就是**如何用有限的记忆、参数或统计量替代这些不可访问的历史数据**。

这里的 $w_i$ 也有意义。每个任务同等重要、每个样本同等重要、近期任务优先，是不同的优化目标。实验里用什么权重，应该和最终评价方式一致。

### 2. 遗忘与可塑性是同时存在的问题

假设模型先学会商品情感分类，再学习新闻主题分类。后者可能改写前者依赖的表示，导致旧任务准确率下降，这就是灾难性遗忘。

但如果为了不忘而冻结所有可训练参数，旧任务表现或许保持了，新任务却学不会。因此持续学习至少同时关心：

- **稳定性（stability）**：保留旧任务和已有能力。
- **可塑性（plasticity）**：能够获得新任务所需的知识。
- **迁移（transfer）**：旧知识帮助新任务，或新任务反过来帮助旧任务。
- **效率**：任务越来越多时，训练时间、参数和存储是否仍然可接受。

“遗忘为零”本身不是完整的成功标准。一个从来没有学会过任务的模型，也可能没有明显遗忘。

### 3. 它和多任务学习、领域适应有什么关系？

| 问题 | 数据访问方式 | 主要关注 |
|---|---|---|
| 多任务学习 | 多个任务数据可以联合访问 | 共享知识，提高总体性能 |
| 持续学习 | 数据按序到达，历史访问受限制 | 新任务学习与历史能力保持 |
| 领域适应 | 从源域迁移到目标域 | 目标域效果，未必要求保留源域 |
| 在线学习 | 样本逐个或小批到达 | 有限访问下的及时更新 |
| 持续测试时适应 | 测试流上进行适应，常无标签 | 分布漂移中的预测和稳定性 |

持续学习不一定是单遍在线训练。许多论文在当前任务上训练多个 epoch，只是禁止任意访问旧任务。反过来，在线更新也不自动意味着评估了所有历史能力。

多任务联合训练常被列作参考上界，因为它能访问全部任务数据；但它不是对任意优化器、预算和任务组合都成立的数学上界。

## 二、从视觉到语言：任务设定如何变化？

### 1. 经典的三种 incremental learning

[Three Scenarios for Continual Learning][1] 对 task-incremental、domain-incremental 和 class-incremental 的区分很适合入门。关键不只是训练时数据怎样切分，还包括**测试时模型知道什么**。

| 设定 | 训练中发生什么 | 测试时的信息 | 例子 |
|---|---|---|---|
| Task-incremental（Task-IL） | 依次学习不同任务 | 提供任务身份或限定输出空间 | 知道当前图片属于哪一组分类任务 |
| Domain-incremental（Domain-IL） | 输入分布变化，输出语义保持 | 通常不提供域身份 | 数字类别不变，图像风格逐步变化 |
| Class-incremental（Class-IL） | 新类别不断加入 | 必须在所有已见类别中区分 | 先学前 10 类，后来扩展到 100 类 |

以 Split CIFAR-100 为例，把 100 类拆成 10 个任务，每个任务 10 类。Task-IL 测试时可以告诉模型“这张图来自第 3 组”，只需在该组类别中选择；Class-IL 则要在全部已见类别中选择。前者成绩更高，并不说明算法更强，因为模型获得了更多信息。

还要区分训练阶段已知任务边界与测试阶段知道 task ID。很多方法训练时按任务新增模块，测试时却使用统一模型或学习到的路由器。

### 2. 为什么视觉论文占据了经典阅读清单？

许多经典持续学习工作在 MNIST、CIFAR、ImageNet 或视觉分类任务流上发展起来。这样做有几个实际原因：标签明确、任务拆分方便、评价统一，而且比较容易控制模型大小与实验预算。

但视觉研究本身也已经发生变化。早期大量实验从随机初始化训练网络；后来越来越多方法使用预训练 ViT 或 CLIP，只训练 prompt、adapter 或 LoRA。这时“保护过去”除了保留已学类别，还包括不破坏预训练模型的通用表示。

因此，读视觉方法时可以借用其机制，却要保留实验边界。例如：

- iCaRL 面向有 exemplar memory 的类增量学习。
- L2P、DualPrompt 面向预训练视觉模型的 prompt 学习。
- InfLoRA 主要面向 exemplar-free 视觉类增量学习。
- KeepLoRA 进一步在视觉语言模型中考察预训练迁移能力。

把它们搬到语言任务流上，通常需要重新设计输出格式、分类或生成接口和评价流程，而不是换一个数据集名字就完成了迁移。

### 3. 语言模型里至少有三种持续学习

**持续预训练（continual pretraining）**：模型按时间或领域继续学习无标注文本。目标可能是吸收新知识、适应新领域或新语言，同时保持原有语言能力。这里常看 perplexity、领域下游任务和通用能力，不一定有清晰的任务标签。

**持续监督微调／指令微调（continual fine-tuning / instruction tuning）**：按顺序学习文本分类、自然语言推理、问答、摘要等任务。这是 O-LoRA 及本系列最直接关注的场景。

**持续对齐（continual alignment）**：偏好、行为规范或反馈分布不断变化，需要更新模型行为。它涉及不同于普通分类准确率的评价问题，不能只用情感分类任务流代表。

此外，RAG 可以通过更新外部知识库改变模型可用的信息，但如果模型参数没有更新，就不是本文主要讨论的参数持续学习。二者可以互补，不能把“检索到了新文档”直接当作参数中的知识已经学会。

语言任务还有一个细节：**task-ID-free 不等于没有任务线索。** 模型即使没有收到整数编号，仍可能从“判断情感”或“回答问题”的自然语言指令中识别任务。这和没有提示、必须在统一标签空间中区分全部类别的视觉 Class-IL 并不完全等价。

## 三、如何衡量学会、忘记和迁移？

### 1. 先保存一张性能矩阵

统一约定 $a_{t,i}$ 是“训练完第 $t$ 个任务后，在任务 $i$ 上的性能”，第一下标表示训练进度，第二下标表示测试任务。有些论文刚好反过来，引用公式前必须先统一。

下面是一条三任务流的示意数据，数值只是用于解释指标：

| 训练进度 | 测试任务 1 | 测试任务 2 | 测试任务 3 |
|---|---:|---:|---:|
| 学完任务 1 | 80 | — | — |
| 学完任务 2 | 70 | 85 | — |
| 学完任务 3 | 65 | 80 | 90 |

对角线表示任务刚学完时的水平，最后一行表示最终保留下来的能力。只保存最后一行，就丢失了研究遗忘和迁移的重要信息。

### 2. AA 与 LA：最后会多少，刚学完时会多少？

最终任务宏平均性能为：

$$
AA=\frac1T\sum_{i=1}^{T}a_{T,i}.
$$

上例为 $(65+80+90)/3=78.33$。

Learning Accuracy 可定义为：

$$
LA=\frac1T\sum_{i=1}^{T}a_{i,i}.
$$

上例为 $(80+85+90)/3=85$。AA 明显低于 LA，意味着任务学会后发生了损失。

这里采用的是每个任务同等权重的宏平均。如果按测试样本数加权，则是：

$$
AA_{\mathrm{weighted}}
=\frac{\sum_i n_i a_{T,i}}{\sum_i n_i}.
$$

两个指标不能混用。例如一个小任务得到 90 分，一个大任务得到 50 分，任务宏平均是 70；样本加权结果会更接近 50。后面会看到，PS-LoRA 附录 A.2 明确写出了样本加权形式，而 O-LoRA 第 4.1.2 节给出任务平均形式。

### 3. BWT 与 FM：遗忘的两个视角

后向迁移：

$$
BWT=\frac1{T-1}\sum_{i=1}^{T-1}(a_{T,i}-a_{i,i}).
$$

上例是 $[65-80+80-85]/2=-10$ 个百分点。负数表示后续任务平均损伤了旧任务；正数表示存在有益的后向迁移。

遗忘量的一种常见定义是：

$$
FM=\frac1{T-1}\sum_{i=1}^{T-1}
\left(\max_{k\in\{i,\ldots,T\}}a_{k,i}-a_{T,i}\right).
$$

它比较历史最好成绩和最终成绩。上例 FM 为 10，但 FM 一般不等于 $-BWT$：若一个旧任务在后续阶段先受益、后来又下降，历史最高点就不在对角线上。

不同论文会改变最大值范围、是否包含最终 checkpoint，以及是否截断负遗忘，因此公式要和代码对齐。$T=1$ 时没有历史任务，BWT 和 FM 不应直接套分母为 $T-1$ 的式子。

### 4. FWT、通用能力和资源指标

前向迁移的一种零样本定义是：

$$
FWT=\frac1{T-1}\sum_{i=2}^{T}(a_{i-1,i}-b_i),
$$

其中 $b_i$ 是未经历这条任务流的基模型在任务 $i$ 上的零样本表现。它回答“先前任务是否让模型在尚未学习的新任务上表现更好”。

也有论文用新任务训练后的成绩、少样本适应效率或不同基线定义迁移；这些都应另行说明。尤其在 LLM 上，$b_i$ 可能很高，不能默认等于随机猜测。

除任务流指标外，还应该报告：

| 指标 | 回答的问题 |
|---|---|
| 预训练／通用能力探针 | 顺序微调后 MMLU、一般语言能力等是否下降？ |
| 学习曲线与训练步数 | 是否只是用更多训练预算换取效果？ |
| 可训练参数 | 每一步需要优化多少参数？ |
| 总持久存储 | 历史 adapter、基矩阵、Fisher、旧模型等占多少空间？ |
| 峰值显存和训练时间 | 中间激活、优化器状态、SVD 等代价有多大？ |
| 推理开销 | 是否要路由、拼接长 prompt、运行多个专家？ |

“每个任务只训练很少参数”不意味着总存储恒定；“rehearsal-free”也不意味着不保存历史统计量。

## 四、方法路线一：Replay，用有限记忆近似旧任务

### 1. Experience Replay：最直接、也很难绕开的基线

如果模型只看新任务，就不知道旧任务的损失正在变大。Replay 因此保存一个小缓冲区 $\mathcal M$，每次把新样本和旧样本混合训练：

$$
L=L_{\mathrm{new}}
+\lambda\,\mathbb E_{(x,y)\sim\mathcal M}\ell(f_\theta(x),y).
$$

这条路线的核心不只有“复习”，还包括保存什么：随机 reservoir sampling 适合未知长度数据流；按类平衡采样可以避免新类别或大任务挤占全部缓存；挑选代表样本则试图用少量数据覆盖更大的分布。

它直接约束旧行为，因此往往是很强的基线。但预算必须说清楚：固定总缓存与每个任务固定缓存不是同一回事，后者会随任务数增长。

### 2. iCaRL 与 DER：记住的也可以是表征和输出

**iCaRL**[2] 将类增量表征学习、旧类别蒸馏和 exemplar 集合结合，并使用 exemplar 的特征均值进行分类。它的贡献是把“记忆”“表示”“分类器”作为一套系统设计，避免只训练新类别时分类器完全偏向近期数据。

**Dark Experience Replay（DER）**[3] 除了保存输入，还保存模型过去产生的 logits，再用 replay 对齐这些输出。它保留的不只是标签答案，还包括模型对不同类别的相对响应。DER++ 进一步结合真实标签监督。代价是输出缓存与样本缓存都要计入资源。

### 3. LAMOL 与 LFPT5：语言模型自己生成复习材料

**LAMOL**[4] 把语言模型同时作为任务求解器和旧任务样本生成器。学习新任务前，通过生成 token 触发旧数据生成，再混合真实新数据和伪旧数据训练。

**LFPT5**[5] 将这一思路用于冻结 T5 的 soft prompt：训练少量 prompt 参数，同时学习求解和生成，以伪样本及蒸馏缓解 few-shot 持续学习中的遗忘。

它们很适合说明视觉经验如何迁移到语言：复习思想不变，但语言模型本身提供了生成记忆的手段。限制也随之变化：伪样本可能丢失少数类别、产生错误标签，长任务流中还可能累积生成偏差。因此“不存真实历史数据”不等于“没有历史数据依赖”。

## 五、方法路线二：正则化，保护重要参数或模型行为

### 1. EWC：哪些参数不能随便动？

EWC[6] 在旧任务收敛点附近，用对角 Fisher 信息近似参数重要性：

$$
L(\theta)=L_t(\theta)
+\frac{\lambda}{2}\sum_jF_j(\theta_j-\theta_j^*)^2.
$$

如果 $F_j$ 大，改变该参数需要支付较大代价；小的重要性则允许更多适应。它可以解释为对旧任务参数后验的局部近似。

但对角近似忽略参数之间的关联；旧任务最优点附近的重要性也未必适用于远处。长任务流中不断累积约束，容易出现“保住了旧参数，却学不动新任务”的问题。Fisher 的估计数据和存储也应该计入成本。

### 2. SI：重要性可以从优化过程里估计

Synaptic Intelligence（SI）[7] 不只在任务结束后估计重要性，而是沿训练轨迹统计参数变化对损失下降的贡献，再据此限制未来变化。

EWC 和 SI 可以放在一起理解：它们都把旧知识压缩成参数级的重要性权重，但压缩依据不同。这里的“重要性”始终是估计量，不是神经网络知识位置的精确标签。

### 3. LwF：参数可以变，输出尽量别变

Learning without Forgetting（LwF）[8] 用旧模型作为教师，在当前输入上保留旧任务输出，然后训练学生兼顾新任务监督与蒸馏：

$$
L=L_{\mathrm{new}}
+\lambda\,D_{\mathrm{KL}}
\bigl(p_{\mathrm{teacher}}(\cdot\mid x)
\|p_{\mathrm{student}}(\cdot\mid x)\bigr).
$$

和参数保护相比，蒸馏更接近功能约束。但如果新输入与旧任务相差很远，旧模型在新数据上的输出未必能代表旧任务分布。它通常不要求旧任务原始数据，却需要教师模型或保存的教师响应。

在语言模型里，蒸馏可以作用在 token 分布、隐藏表示或注意力上；长序列、词表大小和教师前向计算都会增加成本。

## 六、方法路线三：梯度与子空间约束，限制更新方向

### 1. GEM 和 A-GEM：不要沿着伤害旧损失的方向走

GEM[9] 保存 episodic memory，并利用旧样本梯度检查当前更新是否会增加历史损失。设当前梯度为 $g$，旧任务梯度为 $g_i$，采用更新 $-\eta g$ 时，一阶近似为：

$$
\Delta L_i\approx-\eta\langle g_i,g\rangle.
$$

因此 $\langle g_i,g\rangle\geq0$ 是一个局部“不增加旧损失”的条件。GEM 在冲突时寻找接近当前梯度、又满足历史约束的替代梯度。

A-GEM[10] 把多组历史约束简化成一个参考梯度约束，减少计算。二者都提醒我们：梯度约束可以和 replay 共存，方法类别不是互斥的。

局限也很明确：这是当前点附近的一阶判断，依赖 memory 的代表性，也依赖优化器如何把梯度转成实际位移。

### 2. OGD：保存历史输出敏感方向

OGD[11] 将新更新投影到历史模型输出梯度所张成空间的正交补。它和 GEM 的差别是，不只是比较多个损失梯度的符号关系，而是保存输出敏感方向并进行几何约束。

若 $Q$ 的列是历史方向的正交基，可写成：

$$
g_{\mathrm{proj}}=(I-QQ^\top)g.
$$

方向保留越多，保护通常越强，但剩余可学习空间越小。大模型中保存高维梯度基也可能昂贵。

### 3. GPM：用激活来近似需要保护的空间

GPM[12] 对历史层输入或激活做 SVD，按能量阈值保存主要方向。为什么输入特征和梯度有关？对线性层 $e=Wh$：

$$
\frac{\partial L}{\partial W}
=\frac{\partial L}{\partial e}h^\top.
$$

输入 $h$ 决定了梯度的输入侧方向。若 $M$ 保存历史输入基，则可以用：

$$
G_{\mathrm{proj}}=G(I-MM^\top)
$$

限制新梯度。这里用右乘，是因为本文使用列向量输入，且 $G$ 的列对应输入维度。

它比保存全部旧样本或高维梯度更紧凑，但 SVD 保留的是特征能量，不必然等同于旧任务损失敏感度。这个区别后来在正交 LoRA 中仍然重要。

## 七、方法路线四：参数隔离、Prompt 与模块化

### 1. Progressive Networks 与 PackNet：把空间分给不同任务

Progressive Networks[13] 为新任务增加新网络列，旧列冻结，新列通过横向连接复用历史表示。固定旧参数降低了直接覆盖，新增参数提高了可塑性，但模型规模会增长，任务选择也需要处理。

PackNet[14] 则通过训练、剪枝和掩码把已有网络空间分配给任务。它避免每次增加完整网络，但要保存任务掩码，并面临剩余容量耗尽的问题。

两者体现同一个取舍：**把共享参数里的干扰转化成容量分配和任务选择问题。**

### 2. L2P、DualPrompt：预训练模型中的可学习提示

L2P[15] 在冻结的预训练视觉模型上维护 prompt pool，根据输入与 key 的匹配选择提示。它试图通过输入驱动的选择减少对显式 task ID 的依赖。

DualPrompt[16] 将通用提示与任务相关的专家提示区分开，希望同时承载共享知识和任务差异。这里要注意它们最初是视觉模型上的 prompt 方法，不能因为叫 prompt 就直接理解为语言模型里的一句话指令。

冻结 backbone 减少了可训练参数，却没有自动解决遗忘：prompt 本身可能被覆盖，选择机制也可能把样本路由到错误模块。

### 3. Progressive Prompts：语言模型的任务提示链

Progressive Prompts[17] 依次为任务学习新的 soft prompt，冻结历史 prompt，并使新任务利用已有提示积累知识。它在语言任务流中提供了很强的稳定性参照。

但历史提示会带来参数或上下文成本，测试时如何选择任务对应的提示也影响比较公平性。它在已知任务条件下表现好，不等于解决了完全无任务信息的统一推理问题。

这也是为什么读 prompt、adapter、LoRA 论文时，必须问清楚：当前输入最终经过哪些模块，谁决定这些模块，以及模型是否会随任务数增长。

## 八、从 PEFT 到语言模型持续学习

LoRA[18] 写作：

$$
W=W_0+sBA,\qquad
A\in\mathbb R^{r\times d_{\mathrm{in}}},
\quad B\in\mathbb R^{d_{\mathrm{out}}\times r}.
$$

冻结 $W_0$ 可以降低训练成本，但顺序更新同一个 $BA$ 仍然会遗忘；为每个任务新增 LoRA，又会遇到容量增长、合并和任务路由的问题。于是持续 PEFT 可以沿下面几类问题来读：

| 问题 | 代表路线 | 需要检查的代价或假设 |
|---|---|---|
| 旧任务方向怎样表示？ | O-LoRA：LoRA 子空间作为历史更新方向的代理 | 因子代理是否足以保护旧任务行为？ |
| 新任务应该选择哪些低秩方向？ | InfLoRA：历史特征正交补中的固定输入基 | 特征覆盖和视觉 Class-IL 协议 |
| 如何兼顾预训练迁移？ | KeepLoRA：主干主空间与任务特征联合保护 | 预训练主空间是否适合当前模型？ |
| 每层应该分配多少容量？ | OA-Adapter：可训练门控与历史子空间约束 | 容量上限、历史存储和软约束 |
| 是否必须限制权重本身为低秩更新？ | GORP：结合 LoRA 与全参梯度低秩投影 | 训练模块和优化器状态是否公平？ |
| 能否控制方向冲突并合并历史更新？ | PS-LoRA：方向、幅值约束和合并 | 坐标冲突指标与功能保持的关系 |
| 如何维持单一合并 LoRA？ | SLAO：利用因子不对称性进行持续合并 | 合并误差、初始化和固定预算 |

其中 O-LoRA[19] 将 OGD 式的历史方向保护推进到语言模型的低秩适配；InfLoRA[20] 则从固定降维基出发证明局部的子空间更新性质。两者动机相近，但约束对象、参数是否可训练以及理论条件不同，不能混为一谈。

GORP[21]、PS-LoRA[22]、SLAO[23] 也说明近期研究已经不只是“再加一个正交损失”：有人扩大可优化的权重空间，有人控制有效更新的方向和幅值，有人处理历史模块合并后的固定预算。这些方法的细节适合在专题文章展开，这里先明确它们在领域地图中的位置。

## 九、视觉领域常见实验：为什么它们不能直接与语言结果混排？

| 数据集或协议 | 常见构造 | 主要考察 |
|---|---|---|
| Permuted MNIST | 每个任务使用固定但不同的像素排列 | 分布变化中的共享参数学习 |
| Split MNIST | 把数字类别拆成多个任务 | Task-IL 与 Class-IL 的区别 |
| Split CIFAR-100 | 将 100 类拆成若干增量阶段 | 新类别获取、旧类别遗忘和类别混淆 |
| ImageNet 子集、ImageNet-R | 使用不同类别流，或有风格变化的类别数据 | 预训练视觉表示上的持续适应 |
| DomainNet 等多域数据 | 依次学习素描、照片等域 | 域漂移下的知识保持 |
| CORe50 等对象流 | 类别、实例和拍摄条件按不同方式到达 | 更接近连续观察的非独立同分布数据 |

相同数据集名字也不足以复现实验。例如 CIFAR-100 的 10×10 与先学 50 类再逐步扩展不同；固定 memory 总量和固定每类 memory 不同；随机初始化 ResNet 与强预训练 ViT 更不是同一难度。

所以视觉经验最值得借鉴的是实验控制方式：保存完整类别顺序、区分单头和多头、说明 memory 配额、报告每阶段学习曲线。迁移到语言任务流时，这些习惯依然适用。

## 十、语言模型论文反复使用的实验协议

这一节对照了 O-LoRA、OLieRA、GORP、PS-LoRA、SLAO 的实验章节和附录。它们确实属于一条相近的评测传统，但不能把“沿用相同 benchmark”理解成全部配置完全一致。

### 1. Standard CL：五个候选数据集与四任务短序列

常见的基础文本分类数据池包括：

| 数据集 | 任务 | 要注意什么 |
|---|---|---|
| AG News | 新闻主题分类 | 常见版本为 4 类 |
| Amazon Reviews | 商品评论情感／评分分类 | polarity 与 full 版本标签数不同 |
| Yelp Reviews | 商户评论情感／评分分类 | 同样要区分 polarity 与 full |
| DBpedia | 百科主题分类 | 常见版本为 14 类 |
| Yahoo Answers | 问答主题分类 | 这是主题分类，不是生成问答 |

这里有一个非常容易复制错的细节。O-LoRA 第 4.1.1 节介绍了五个数据集，但附录 A.3、Table 7 的短序列只有四个任务：

| 序列 | O-LoRA 附录列出的实际顺序 |
|---|---|
| Order-1 | DBpedia → Amazon → Yahoo → AG News |
| Order-2 | DBpedia → Amazon → AG News → Yahoo |
| Order-3 | Yahoo → Amazon → AG News → DBpedia |

Yelp 出现在数据池和长序列中，但不在这三条短序列里。OLieRA 的附录任务表、SLAO 的 Table 17 也列出相同的四任务短序列；PS-LoRA 则在主表直接标注 **Standard ($N=4$)**。GORP 第 4.1 节也明确提到四任务设定。

因此写实验不能只写“使用标准五任务 benchmark”，而应该列出实际任务清单。使用包含 Yelp 的五任务流当然可以，但应明确说明这和上述四任务短序列的区别。

### 2. Long / Large Number of Tasks：十五任务流

O-LoRA 沿用 Progressive Prompts 的长序列框架，将任务扩展到：

| 来源 | 数据集 | 主要能力 |
|---|---|---|
| 基础 CL 数据池 | Yelp、Amazon、DBpedia、Yahoo、AG News | 情感和主题分类 |
| GLUE 子集 | MNLI、QQP、RTE、SST-2 | 自然语言推理、问题复述、情感分类 |
| SuperGLUE 子集 | WiC、CB、COPA、MultiRC、BoolQ | 词义、推理、因果选择、阅读理解 |
| 独立电影评论数据 | IMDB | 情感分类 |

IMDB 不是 SuperGLUE 的组成部分，虽然部分论文附录表格把它排在同一类别下。QQP 是问题对是否语义重复的判断，也不是“段落检测”。

O-LoRA 附录列出的三种长序列如下；其中 AG 是 AG News，BoolQA 指 BoolQ：

- **Order-4**：MNLI → CB → WiC → COPA → QQP → BoolQ → RTE → IMDB → Yelp → Amazon → SST-2 → DBpedia → AG News → MultiRC → Yahoo。
- **Order-5**：MultiRC → BoolQ → WiC → MNLI → CB → COPA → QQP → RTE → IMDB → SST-2 → DBpedia → AG News → Yelp → Amazon → Yahoo。
- **Order-6**：Yelp → Amazon → MNLI → CB → COPA → QQP → RTE → IMDB → SST-2 → DBpedia → AG News → Yahoo → MultiRC → BoolQ → WiC。

为什么需要多种顺序？相近任务连续出现可能产生迁移，差异很大的任务连续出现则可能造成较强干扰；一个方法的优势也可能依赖哪些任务放在最后。

这仍然主要是有限答案空间的语言理解任务流。它比四任务短序列更长，但不能仅凭这组结果声称模型能持续获得代码、数学和自由生成能力。

### 3. 指令模板与评价：为什么 GLUE 的名字不等于 GLUE 官方分数？

这类论文常把任务统一为自然语言输入和文本答案。例如：

> 判断下面评论的情感，从给定选项中选择答案。
>
> 评论：……
>
> 答案：positive

同一个任务可以用自由生成、候选标签概率比较或分类头评估，结果未必相同。还需要约定大小写、标点、标签同义词、非法输出和多个 token 的标签概率如何处理。

O-LoRA 的附录任务表统一列 Accuracy。它借用 GLUE、SuperGLUE 数据集，不意味着复用了完整官方排行榜的聚合方式；MultiRC 等任务的官方评价也不只有简单 accuracy。

所以必须记录**实际任务重写、label verbalizer、答案解析和评价函数**。评测脚本在这里和算法本身一样重要。

### 4. 采样预算：有共同传统，但不是一套万能数字

O-LoRA 在长序列段落写明每任务随机选取 1,000 个训练样本、每类留出 500 个验证样本；GORP 附录 B 对长序列也写到 1,000 个训练样本，并涉及每类 500 个验证和测试样本；SLAO 第 5.1 节沿用了类似采样描述。

这些文字不能直接推出所有数据集都有这么多可用样本。例如 CB、COPA 等小数据集可能不足以满足某个统一配额。复现时必须检查采样代码究竟是取上限、使用全部可用数据，还是有额外处理，并确保 train/validation/test 不重叠。

还要区分“每任务 1,000”与“每类 1,000”。后者会让不同类别数的任务拥有完全不同的训练预算。实际划分文件、采样种子和样本 ID 比一句“following O-LoRA”更可靠。

### 5. 相同任务表，也可能有不同训练配置

| 论文与位置 | 明确可见的协议特征 | 比较时的提醒 |
|---|---|---|
| O-LoRA §4.1、附录 A.1–A.3 | 短序列与 15 任务长序列；T5 实验 1 epoch、学习率 $10^{-3}$ | 附录区分具体任务顺序和正则系数 |
| OLieRA §4.1、附录 A.1 | 延续相似任务流；T5 用 2 epochs，短序列学习率 $10^{-3}$、长序列 $5\times10^{-4}$ | 不能因为任务相同就说训练预算完全相同 |
| GORP §4.1、附录 B–D | T5-Large、LLaMA2；LoRA 与部分全参训练结合 | T5 中不同模块使用不同学习率与参数化 |
| PS-LoRA §4、附录 A.2、F.1 | 明确 4/15 任务；附录写样本加权准确率；部分误差条来自三种顺序 | 不能把跨顺序方差解释成固定顺序的跨 seed 方差 |
| SLAO §5.1、附录 B.1、B.12 | 4/15 任务传统及 SuperNI；Llama/Qwen；每顺序 3 seeds | 附录给出 Llama 配置，例如 rank 8、query/value 模块 |

这个表的目的不是挑出一份“正确配置”让所有工作照抄，而是说明：**任务清单、采样、主干、训练预算、评价口径共同定义了一次可比较的实验。**

### 6. SuperNI：从分类走向生成任务

Super-NaturalInstructions[24] 是包含专家指令的多任务集合，不是一条唯一固定的持续学习任务流。研究者会从中选任务，再构造顺序。

SLAO 沿 SAPT 的选择方式使用 15 个任务，涵盖：

- 对话生成，例如 task639、task1590、task1729。
- 信息抽取，例如 task181、task748、task1510。
- 问答，例如 task002、task073、task591。
- 摘要，例如 task511、task1290、task1572。
- 情感分类，例如 task363、task875、task1687。

SLAO 第 5.1 节报告每任务 1,000 个训练实例，以及验证／测试侧 100 个实例的设置；具体划分仍应以数据文件为准。它的 Table 16 对生成任务使用 ROUGE-L，对最后三项分类任务使用 Accuracy，Table 17 列出两种顺序。

这比纯分类更接近多技能持续适应，但混合指标的平均值应叫平均任务分数，不能全部叫“准确率”。ROUGE-L 也主要反映参考文本重合，不等于事实正确性或完整任务质量。

### 7. TRACE：更异质的 LLM 能力流

TRACE[25] 专门面向语言模型持续学习，包含 C-STANCE、FOMC、MeetingBank、Py150、ScienceQA、NumGLUE-cm、NumGLUE-ds、20Minuten 等任务，覆盖立场或金融分类、会议摘要、代码、科学问答、数值推理和多语言文本简化。

PS-LoRA 使用 TRACE 进行额外验证；GORP 附录也讨论了这一类更异质的任务。这里不同任务使用不同评价函数，例如 Accuracy、ROUGE-L、代码相似度和 SARI，不能拿单个聚合分数与 T5 四任务分类 AA 直接比较。

TRACE、SuperNI 与 4/15 分类任务流可以构成不同层次的证据：前者扩展能力范围，后者保留与历史方法的可比性。它们并不互相替代。

### 8. 通用能力保持：旧下游任务之外还要测什么？

O-LoRA 第 4.2 节还研究了经过 Alpaca 指令微调的 LLaMA，在经历 CL 任务流后的零样本 MMLU 表现。这补上了一个单看旧任务准确率看不到的问题：模型可能记住所有分类任务，却丢失预训练阶段的通用能力。

做这类评测应说明基准点是原始 base、instruction-tuned 模型还是任务 1 checkpoint；保持 zero-shot/few-shot 提示和答案解析一致，并避免用保留测试集反复调参。

MMLU 只是一个探针，不代表全部通用能力。依据研究目标，还可以增加语言建模、代码、数学、生成质量等评价，但最好先明确每个探针要检验的具体主张。

## 十一、读一篇论文，应该形成什么判断？

看完方法和实验之后，我希望能用下面几个问题概括它：

1. **问题是什么？** 视觉类别增量、域漂移、文本分类流，还是生成技能流？
2. **信息条件是什么？** 可以 replay 吗？训练和测试分别知道哪些任务信息？
3. **保护对象是什么？** 样本、logits、重要参数、梯度、特征基、LoRA 因子，还是完整权重更新？
4. **稳定性从哪里来？** 是否伴随 LA 下降、参数增长、更多训练或更强先验？
5. **比较是否同口径？** 主干、初始化、数据划分、任务顺序、学习率与评价公式是否对齐？
6. **证据覆盖到哪里？** 只验证四任务分类，还是也覆盖长序列、生成能力和预训练知识保持？

这一领域的演进可以这样理解：早期研究集中于“怎样让网络不忘旧任务”；预训练模型普及后，又要回答“怎样利用强主干而不破坏它”；PEFT 把成本降下来之后，研究继续追问“少量参数到底承载了哪些知识，应该共享、隔离还是合并”。

沿着这条线，正交 LoRA 就不再是突然出现的一组技巧，而是从历史梯度保护、特征子空间约束和参数高效适应发展出来的一条路线。[下一篇](/posts/orthogonal_lora)会逐篇讲它们的 motivation 和 methodology，并进一步引出：用一个低秩因子作为过滤器，究竟在什么条件下能够控制完整更新？

## 参考文献与泛读顺序

建议先读 [1] 理解任务设定，再从 [2]–[17] 中按方法路线各选一到两篇；进入语言方向后，读 [4]、[5]、[17]、[19]，并对照 [21]–[25] 的实验部分。无需第一次就把每篇的所有推导读完，但应能说明它保护什么、保存什么、测试时知道什么。

1. [Three Scenarios for Continual Learning](https://arxiv.org/abs/1904.07734)：Task-IL、Domain-IL、Class-IL 的区分。
2. [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)：类增量、exemplar 与蒸馏。
3. [Dark Experience for General Continual Learning](https://arxiv.org/abs/2004.07211)：DER 与输出 replay。
4. [LAMOL: Language Modeling for Lifelong Language Learning](https://arxiv.org/abs/1909.03329)：语言模型生成式 replay。
5. [LFPT5](https://arxiv.org/abs/2110.07298)：few-shot 语言持续学习与 prompt。
6. [Overcoming Catastrophic Forgetting in Neural Networks](https://doi.org/10.1073/pnas.1611835114)：EWC。
7. [Continual Learning Through Synaptic Intelligence](https://arxiv.org/abs/1703.04200)：SI。
8. [Learning without Forgetting](https://arxiv.org/abs/1606.09282)：输出蒸馏。
9. [Gradient Episodic Memory for Continual Learning](https://arxiv.org/abs/1706.08840)：GEM。
10. [Efficient Lifelong Learning with A-GEM](https://arxiv.org/abs/1812.00420)：简化梯度约束。
11. [Orthogonal Gradient Descent for Continual Learning](https://proceedings.mlr.press/v108/farajtabar20a.html)：OGD。
12. [Gradient Projection Memory for Continual Learning](https://arxiv.org/abs/2103.09762)：GPM。
13. [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)：网络扩展与历史复用。
14. [PackNet](https://arxiv.org/abs/1711.05769)：剪枝与任务参数分配。
15. [Learning to Prompt for Continual Learning](https://arxiv.org/abs/2112.08654)：L2P。
16. [DualPrompt](https://arxiv.org/abs/2204.04799)：通用与专家提示。
17. [Progressive Prompts: Continual Learning for Language Models](https://arxiv.org/abs/2301.12314)：语言模型提示链和长任务流。
18. [LoRA](https://arxiv.org/abs/2106.09685)：低秩适配基础。
19. [Orthogonal Subspace Learning for Language Model Continual Learning](https://aclanthology.org/2023.findings-emnlp.715/)：O-LoRA，尤其 §3.2、§4.1、附录 A；[官方实现](https://github.com/cmnfriend/O-LoRA)。
20. [InfLoRA](https://arxiv.org/abs/2404.00228)：固定输入基与视觉 Class-IL。
21. [Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](https://arxiv.org/abs/2507.02503)：GORP，§4.1、附录 B–D。
22. [Resolving Conflicts in Lifelong Learning via Aligning Updates in Subspaces](https://arxiv.org/abs/2512.08960)：PS-LoRA，§4、附录 A.2、F。
23. [Merge before Forget: A Single LoRA Continual Learning via Continual Merging](https://arxiv.org/abs/2512.23017)：SLAO，§5.1、附录 B.1、B.12。
24. [Super-NaturalInstructions](https://arxiv.org/abs/2204.07705)：指令任务集合。
25. [TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models](https://arxiv.org/abs/2310.06762)：异质语言能力流。
26. [Orthogonal Low-rank Adaptation in Lie Groups for Continual Learning of Large Language Models](https://arxiv.org/abs/2509.06100)：OLieRA，§4.1、附录 A。
27. [A Continual Learning Survey: Defying Forgetting in Classification Tasks](https://arxiv.org/abs/1909.08383)：补充视觉分类持续学习的系统背景。
28. [Continual Learning of Large Language Models: A Comprehensive Survey](https://arxiv.org/abs/2404.16789)：补充 LLM 持续学习的更广问题分类。

[1]: https://arxiv.org/abs/1904.07734
[2]: https://arxiv.org/abs/1611.07725
[3]: https://arxiv.org/abs/2004.07211
[4]: https://arxiv.org/abs/1909.03329
[5]: https://arxiv.org/abs/2110.07298
[6]: https://doi.org/10.1073/pnas.1611835114
[7]: https://arxiv.org/abs/1703.04200
[8]: https://arxiv.org/abs/1606.09282
[9]: https://arxiv.org/abs/1706.08840
[10]: https://arxiv.org/abs/1812.00420
[11]: https://proceedings.mlr.press/v108/farajtabar20a.html
[12]: https://arxiv.org/abs/2103.09762
[13]: https://arxiv.org/abs/1606.04671
[14]: https://arxiv.org/abs/1711.05769
[15]: https://arxiv.org/abs/2112.08654
[16]: https://arxiv.org/abs/2204.04799
[17]: https://arxiv.org/abs/2301.12314
[18]: https://arxiv.org/abs/2106.09685
[19]: https://aclanthology.org/2023.findings-emnlp.715/
[20]: https://arxiv.org/abs/2404.00228
[21]: https://arxiv.org/abs/2507.02503
[22]: https://arxiv.org/abs/2512.08960
[23]: https://arxiv.org/abs/2512.23017
[24]: https://arxiv.org/abs/2204.07705
[25]: https://arxiv.org/abs/2310.06762

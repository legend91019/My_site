---
title: "正交 LoRA 真能防止遗忘吗：从子空间复用到双线性优化失稳"
summary: "顺着五篇近期工作，理解正交子空间方法从全参投影、预算分配、乘性几何到双线性失稳的关键假设。"
date: 2026-07-11
tags: ["大模型","持续学习","LoRA","正交子空间"]
category: "论文解读"
draft: false
---

上一篇最后我留下了一个问题：O-LoRA 把不同任务的 LoRA 子空间分开，真的就能防止遗忘吗？

直觉上，这个答案很诱人。旧任务占一个方向，新任务走正交补，大家井水不犯河水。可读完最近几篇工作之后，我现在更愿意把它改写成一个更谨慎的问题：

> 正交到底是在保护知识，还是在消耗模型未来可学习的空间？

这篇不是再讲持续学习入门。默认读者已经知道 Transformer，也已经读过上一篇里关于 replay、prompt、LoRA 和 O-LoRA 的基本铺垫。我会按一个研究链条深读五篇论文：

1. **Sculpting Subspaces**：如果不做 LoRA，而是全参数微调，能不能把更新投到旧知识的正交补里？
2. **OA-Adapter**：如果每层、每个任务复杂度不同，正交空间预算该不该自适应？
3. **OLieRA**：如果 LoRA 的加性更新破坏参数几何，乘性更新会不会更自然？
4. **ELLA**：如果严格正交会阻断迁移，能不能只去相关高能历史方向？
5. **SFOR/BOD 作者稿**：如果 LoRA 本身是双线性参数化，只约束一个因子会不会让优化器从另一个因子逃逸？

先把公平性警告放在前面：这些论文的数据集、backbone、是否 replay、是否参数增长、是否 task-ID-free、指标定义都不完全一致。后面的数字只能在各自实验协议里理解，不能直接排成“谁第一谁第二”的总榜。

## 一、Sculpting Subspaces：全参数微调也能走正交补吗？

第一篇的问题很直接：既然 PEFT/LoRA 的表达能力和参数增长都有代价，能不能保留 **full fine-tuning** 的表达力，同时让更新避开旧任务关键方向？

### 问题与直觉

论文研究 rehearsal-free 的 LLM continual learning。任务流 $\{D_1,\dots,D_T\}$ 顺序到达，学习 $D_t$ 时不重新联合训练历史数据，也不为每个任务新增 adapter。它的核心直觉是：

> 每层权重的高奇异值方向更像旧知识和高曲率方向；新任务更新应该尽量走到这些方向的正交补里。

这和 O-LoRA 的差异很明显：O-LoRA 在低秩 LoRA 子空间里隔离任务；Sculpting Subspaces 直接对全模型权重做 SVD，然后投影梯度。它不是 gradient replay；但层重要性 $I^{(l)}$ 这一项需要在任务边界拿到上一任务样本，或者提前缓存好的统计量，所以它的隐私和存储假设要比“纯梯度投影”更重一点。

### 核心公式

设模型参数为：

$$
\theta=\{W^{(1)},W^{(2)},\dots,W^{(L)}\},\quad
W^{(l)}\in\mathbb{R}^{d_O^{(l)}\times d_I^{(l)}}.
$$

对每层权重做 SVD：

$$
W^{(l)}=U^{(l)}\Sigma^{(l)}(V^{(l)})^\top.
$$

论文用上一任务样本估计层重要性：

$$
I^{(l)}=\frac{1}{N}\sum_{i=1}^{N}\cos(X_i^{(l)},Y_i^{(l)}),
\quad Y_i^{(l)}=W^{(l)}X_i^{(l)}.
$$

所以它虽然不把旧样本混进新任务的梯度训练，并不等于完全不依赖历史数据。任务切换时仍要访问上一任务样本来计算 $I^{(l)}$，或者提前缓存足够的层统计量；这项存储与隐私假设需要单独记账。

然后把 $I^{(l)}$ 归一化，并决定每层保留比例：

$$
r^{(l)}=mrr+I^{(l)}(trr-mrr),
$$

其中实现里使用过 $mrr=0.1,trr=0.8$。梯度投影写作：

$$
\nabla_{\text{proj}}^{(l)}
=
\nabla_{W}^{(l)}
-U_{\text{high}}^{(l)}(U_{\text{high}}^{(l)})^\top
\nabla_W^{(l)}
V_{\text{high}}^{(l)}(V_{\text{high}}^{(l)})^\top.
$$

这项减掉的是同时落入旧任务高奇异值左、右子空间的更新分量。

理论上，它用旧任务损失的二阶近似：

$$
\Delta L_k\approx \frac{1}{2}\Delta\theta^\top H_k\Delta\theta,
$$

假设旧任务学完后接近局部最优，即 $\nabla L_k(\theta^{(k)})\approx 0$，再用 block-diagonal Hessian 近似和权重 SVD 近似高曲率方向。论文给出的遗忘上界层级可以概括为：

$$
\frac{1}{2}\max_l\lambda_{r^{(l)}+1}^{(l)}c
<
\frac{1}{2}\max_l\lambda_{r+1}^{(l)}c
\le
\frac{1}{2}\lambda_{\max}(H_k)c.
$$

我读这里时会加一个注脚：equal-norm 假设主要是为了理论隔离变量，不是说实际训练里这些更新真的同范数。

### 训练流程与实验设置

流程是：每个新任务开始时，用上一任务数据估计 $I^{(l)}$；对目标矩阵做 SVD；按 $r^{(l)}$ 保留 high-rank 方向；每个 batch 反传后投影梯度；再用投影梯度做全参数更新。

主实验模型包括 T5-Large 770M 和 LLaMA-2 7B；TRACE 用 LLaMA-2-7B-Chat。T5-Large 使用 1 张 H100，全精度，AdamW，学习率 $5\times10^{-5}$，总 batch size 8，1 epoch/task，每个分类数据集按类采样 1000 例。LLaMA-2 7B 使用 8 张 H100、DeepSpeed stage 2、gradient checkpointing，有效 batch 8，AdamW 学习率 $10^{-5}$、weight decay 0.01。

数据包括标准分类任务 AG News、Amazon Reviews、Yelp Reviews、DBpedia、Yahoo Answers；15-task benchmark 加入 GLUE、SuperGLUE 和 IMDB；TRACE 覆盖多语理解、领域知识、算术推理和代码。

baselines 包括 SeqFT、SeqLoRA、IncLoRA、Replay、EWC、LwF、L2P、LFPT5、O-LoRA、SLERP、TIES、ProgPrompt、PerTaskFT、MTL。指标主要是 AA；TRACE 还报告 BWT；通用能力表报告 MMLU、GSM、BBH、TydiQA、BoolQA、PIQA。

### 关键结果与限制

| 设置 | 主要结果 | 我会怎么读 |
|---|---:|---|
| T5-Large 5-task | Adaptive SVD `75.9%`，O-LoRA `75.8%`，LFPT5 `72.7%`，MTL `80.0%` | 和 O-LoRA 接近，但不是同一类成本 |
| T5-Large 15-task | Adaptive SVD `71.3%`，O-LoRA `69.6%`，LFPT5 `69.2%`，ProgPrompt `77.9%`，PerTaskFT `78.1%` | 优于若干 continual baselines，但 ProgPrompt/PerTaskFT/MTL 约束更强 |
| TRACE / LLaMA-2-7B-Chat | Adaptive SVD AA `48.4%`，BWT `7.1%`；O-LoRA `41.3%/6.2%`；SeqFT `23.0%/-8.3%`；MTL `52.3%` | 异质任务上全参投影很有吸引力 |
| 通用能力 | MMLU `46.6 -> 47.7`，GSM `26.1 -> 7.7`，BBH `40.2 -> 34.2`，TydiQA `23.5 -> 35.8` | 不能写成所有能力都提升，GSM/BBH 明显下降 |

消融里，LLaMA-2 7B 标准 5-task 的 Adaptive SVD 为 `79.6%`；把 `mrr/trr` 减半后降到 `51.5%`；移除投影后降到 `31.2%`。这说明 rank 估计和投影不是装饰，是方法的骨架。

它的代价也清楚：反复 SVD 有成本；rank 估计敏感；预分配正交补在长 horizon 可能耗尽；全参数微调和 PEFT 的训练成本不是一回事。

过渡到下一问：如果全参 SVD 可以按层决定保护比例，那么 adapter/LoRA 这类小模块是不是也应该有动态预算？

## 二、OA-Adapter：正交空间预算应该固定吗？

OA-Adapter 接的是另一个现实问题：固定 rank 很粗糙。不同任务、不同层需要的适配维度不一样，为什么每个 adapter 都用同样的 bottleneck？

### 问题与直觉

论文目标是在一个 end-to-end 训练阶段里，同时做动态预算分配和历史子空间正交保护。它不想先调任务目标、再单独压缩预算，因为这种多阶段流程可能让优化目标和预算分配错位。

它的直觉是：给 adapter bottleneck 的每个潜在维度加一个可训练开关。被打开的维度参与当前任务；历史任务只记录已激活的上投影子空间；新任务只和这些历史激活方向做正交。

### 核心公式

标准 adapter 可写成：

$$
y=x+W_2f(W_1x+b_1)+b_2,
\quad
W_1\in\mathbb{R}^{r\times d},\quad
W_2\in\mathbb{R}^{d\times r}.
$$

OA-Adapter 改成：

$$
y=x+W_2\Gamma W_1x,\quad
\Gamma=\operatorname{diag}(\gamma).
$$

其中：

$$
\gamma_i=\operatorname{soft}(g_i;\tau)
=\operatorname{sign}(g_i)\max(|g_i|-\tau,0).
$$

$g$ 和阈值 $\tau$ 都可训练。有效 rank 是：

$$
r_{\text{eff}}=\|\gamma\|_0.
$$

任务流目标可以写成最大化顺序到达数据上的条件对数似然：

$$
\max_{\phi}\sum_{t=1}^{T}
\sum_{(x_i^t,y_i^t)\in D_t}
\log P_{\phi}(y_i^t\mid x_i^t).
$$

从分解角度看：

$$
W_2\Gamma W_1
=
\sum_{i=1}^{r_{\max}}\gamma_i W_2[:,i]\otimes W_1[i,:].
$$

第 $k$ 个任务的 adapter 更新为：

$$
\Delta W_k
=W_2^{(k)}\Gamma^{(k)}W_1^{(k)}
=\tilde W_2^{(k)}W_1^{(k)}.
$$

这里 $\tilde W_2^{(k)}$ 表示被 $\Gamma^{(k)}$ 激活后的上投影列，也就是只保留非零 $\gamma_i$ 对应的 $W_2$ 列及其缩放。

正交约束只作用在当前任务上投影列和历史已激活列之间：

$$
\langle W_2^{(t)}[:,i],\tilde W_2^{(s)}[:,j]\rangle=0,\quad s<t.
$$

对应损失：

$$
L_{\text{orth}}^{(s,t)}
=
\sum_{i,j}\langle W_2^{(t)}[:,i],\tilde W_2^{(s)}[:,j]\rangle^2,
$$

总损失为：

$$
L_{\text{total}}
=L_{\text{task}}^{(t)}
+\lambda_{\text{orth}}\sum_{s<t}L_{\text{orth}}^{(s,t)}.
$$

### 训练流程与实验设置

base model 冻结，每层插入 OA-Adapter。每个任务训练当前 adapter 的 $W_1,W_2,g,\tau$；任务结束后冻结历史激活子空间，后续任务对这些子空间做正交正则。

主结果用 T5-large 770M；规模表还包括 T5-base 220M、T5-XL 3B、Llama-7B。T5 实验使用 4 张 RTX 3090 + DeepSpeed，1 epoch，batch size 32，dropout 0.1，无 weight decay。主要 bottleneck budget 为 16，阈值从 $\{10^{-3},10^{-4},10^{-5}\}$ 搜索。

Standard CL 有三种顺序：DBpedia/Amazon/Yahoo/AG 等排列；Long Sequence 是 15 个 GLUE、SuperGLUE、IMDB 和分类任务；SuperNI 覆盖 dialogue generation、information extraction、QA、summarization、sentiment analysis。baselines 包括 Replay、L2P、LFPT5、O-LoRA、ProgPrompt；指标是 AA、FWT、BWT。

### 关键结果与限制

| 设置 | OA-Adapter | 对照 |
|---|---:|---|
| T5-large Standard CL | AA/FWT/BWT `76.0/-2.7/-7.5` | O-LoRA `75.3/-3.6/-9.1`，ProgPrompt `75.1/-2.3/-8.1` |
| Long Sequence | `69.2/-4.4/-3.2` | O-LoRA `68.7/-6.2/-4.1` |
| SuperNI | `29.3/-5.2/-6.0` | O-LoRA `25.9/-7.8/-24.6`，Replay `20.5/-1.4/-15.8` |
| 参数效率 budget 16 | OA `1.96M/76.0` | O-LoRA `4.72M/75.3`，参数少 `58.5%` |

消融很有信息量：去掉正交后，Standard/Long/SuperNI 变成 `57.1/52.8/13.7`；固定预算变成 `73.3/65.6/24.2`。这说明正交是主贡献，预算自适应是额外增益。动态阈值也稳定优于固定阈值。

限制是：SuperNI 绝对分数仍低；历史子空间和 pairwise 正交计算随任务数增长；主方法无 replay，但每个任务仍有 task-specific adapter 管理；训练需要任务边界。它解决的是“预算该如何分配”，不是“正交是否会从根上失稳”。

过渡到第三问：如果 adapter/LoRA 的子空间已经可以自适应了，那更新形式本身有没有问题？加性更新 $W+\Delta W$ 是否保留了预训练权重的几何？

## 三、OLieRA：LoRA 的加性更新是否太线性了？

OLieRA 把视角从“空间分配”推到“参数几何”。它认为 O-LoRA、N-LoRA 这类方法虽然用正交减少干扰，但仍然把更新写成 $W\rightarrow W+\Delta W$，这可能扭曲预训练参数的内在结构。

### 问题与直觉

论文想保留 O-LoRA 的 replay-free、parameter-efficient 和 task-ID-free inference，同时让更新在 Lie group / Lie algebra 框架中发生。直觉是：

> 在 Lie algebra 里学习低秩扰动，再通过 exponential map 以乘性方式回到权重空间。

这让它不只约束 LoRA 的某个因子，而是对 full task-update 子空间施加正交约束。

### 核心公式

普通 LoRA：

$$
W+\Delta W=W+BA,\quad
B\in\mathbb{R}^{d\times r},\quad
A\in\mathbb{R}^{r\times k}.
$$

OLieRA 定义 Hadamard 乘法下的 Lie group：

$$
G=\{W\in\mathbb{R}^{b_1\times\cdots\times b_k}\mid W[a_1,\dots,a_k]\neq 0\}.
$$

乘性更新是：

$$
W\rightarrow W\odot\exp(\Delta W),\quad \Delta W=BA.
$$

一阶 Taylor 展开：

$$
\exp(\Delta W)=I+\Delta W+o(\|\Delta W\|),
$$

这里的 $I$ 不是矩阵乘法里的 identity matrix，而是 Hadamard 乘法下的全 1 neutral element，满足 $W\odot I=W$。二阶和一般 $n$ 阶展开分别是：

$$
\exp(\Delta W)
=I+\Delta W+\frac{1}{2}\Delta W^{\odot 2}
+o(\|\Delta W^{\odot 2}\|),
$$

$$
\exp(\Delta W)
=I+\Delta W+\frac{1}{2!}\Delta W^{\odot 2}
+\cdots+\frac{1}{n!}\Delta W^{\odot n}
+o(\|\Delta W^{\odot n}\|),
$$

其中 $\Delta W^{\odot n}$ 表示逐元素 Hadamard 幂。

所以：

$$
W\odot\exp(\Delta W)
\approx
W+W\odot(BA).
$$

它的正交损失近似写为：

$$
L_{\text{orth}}
=
\sum_{i\neq j}\|\exp(\Delta W_i)\exp(\Delta W_j)^\top\|_F
\approx
\sum_{i\neq j}\|(I+B_iA_i)(I+B_jA_j)^\top\|_F.
$$

总损失：

$$
L_{\text{total}}
=L_{\text{task}}+
\lambda\sum_{i\neq j}\|(I+B_iA_i)(I+B_jA_j)^\top\|_F.
$$

论文还用 Fisher 探针看冲突能量：

$$
F_i\approx
\mathbb{E}_{x\sim D}\left[
\left(\partial_{\theta_i}\log p(y|x;\theta)\right)^2
\right],
\quad
E=\sum_iF_i(\Delta\theta_i)^2.
$$

### 训练流程与实验设置

base Transformer 冻结；每个新任务增量学习一个 LoRA 模块；LoRA 更新通过 exponential map / Taylor 近似；学完任务后固定该任务 LoRA；当前任务与历史 task-update 子空间做正交正则。

主结果使用 T5-large 770M；附表有 LLaMA-7B。T5 实验用 A100-SXM4-80GB + DeepSpeed，2 epochs；Standard CL 学习率 $10^{-3}$，Long Sequence 学习率 $5\times10^{-4}$，dropout 0.1，weight decay 0。

数据顺序沿用 Standard CL 的 DBpedia、Amazon、Yahoo、AG 三种 order；Long Sequence 是 15 个 GLUE/SuperGLUE/IMDB/分类任务。baselines 包括 ProgPrompt、PerTaskFT、MTL、SeqFT、SeqLoRA、IncLoRA、Replay、EWC、LwF、L2P、LFPT5、O-LoRA、N-LoRA。

### 关键结果与限制

| 设置 | OLieRA | 对照 |
|---|---:|---|
| T5-large Standard CL avg | `79.6` | N-LoRA `78.8`，O-LoRA `75.8`，MTL `80.0` |
| T5-large Long Sequence avg | `72.6` | N-LoRA `72.4`，O-LoRA `69.6`，PerTaskFT `78.1`，ProgPrompt `77.9` |
| LLaMA-7B avg | `77.7` | N-LoRA `77.6`，O-LoRA `76.1` |
| No LieGroup Mult 消融 | OLieRA `79.9/79.5/79.5` | 去乘性更新 `77.4/77.2/76.9` |

Taylor 阶数消融里，一阶、二阶、三阶差距很小：比如 Order 1 为 `79.4/79.9/79.9`。Fisher energy 表很有意思：O-LoRA 是 `0.12/0.09/0.42`，N-LoRA 近乎 $10^{-10}$，OLieRA 是 `1.04/1.43/3.92`。论文的解释是 OLieRA 不是追求零冲突，而是在敏感方向允许受控更新。

这点对我很关键：**越正交不一定越好**。如果一个方法把所有共享方向都压到零，它可能减少遗忘，也可能损害迁移。

限制也在这里。Hadamard Abelian Lie group 是否真实刻画 LLM 参数几何，并没有被彻底证明；长序列相对 N-LoRA 的提升很小；超参仍可能受任务顺序影响；而且它没有报告 TRACE。

过渡到第四问：如果严格正交不总是好，能不能只保护历史高能方向，把低能方向留给迁移？

## 四、ELLA：能不能只去相关高能历史方向？

ELLA 是我读起来最像“对严格正交做减法”的一篇。它不想把新任务完全推到历史子空间的正交补，而是问：历史更新里哪些坐标真的重要？

### 问题与直觉

ELLA 研究 rehearsal-free、task-agnostic inference 的 LoRA continual learning。它认为硬正交会逐步消耗剩余自由度，也会阻断有用的低能共享表征；复杂 fusion 或 replay 又带来额外存储和结构成本。

它的做法是维护历史 LoRA 更新的聚合矩阵 $W_{\text{past}}$，对新任务更新 $\Delta W_t$ 加一个 Hadamard 能量惩罚：

> 高能历史坐标不要再动太多；低能残余方向仍可复用。

### 核心公式

LoRA 更新：

$$
\Delta W=AB,\quad
A\in\mathbb{R}^{d\times r},\quad
B\in\mathbb{R}^{r\times k}.
$$

历史聚合：

$$
W_{\text{past}}=\sum_{i=1}^{t-1}\Delta W_i.
$$

ELLA 正则项：

$$
L_{\text{ELLA}}
=
\|\Delta W_t\odot W_{\text{past}}\|_F^2.
$$

为避免符号混乱，我把训练目标写成最小化 negative log-likelihood 加惩罚。定义

$$
L_{\text{NLL}}^{(t)}
=
-\sum_{(x,y)\in D_t}\log p_\theta(y\mid x),
$$

则

$$
\min_{\theta}\quad
L_{\text{NLL}}^{(t)}+\lambda L_{\text{ELLA}}.
$$

等价地，也可以写成最大化 log-likelihood 减去 $\lambda L_{\text{ELLA}}$。我在这里不用源文抽取里的 printed signs，因为它们把 log-likelihood 和 penalty 的符号混在了一起，按最小化损失来读并不一致。

理论近似里，定义能量矩阵：

$$
E_{ij}=|(W_{\text{past}})_{ij}|+\epsilon.
$$

考虑正则化问题：

$$
\min_{\Delta W_t}
\frac{1}{2}\|\Delta W_t-G\|_F^2
+\frac{\lambda}{2}\|E\odot\Delta W_t\|_F^2,
$$

其闭式解是逐坐标 shrinkage：

$$
(\Delta W_t^\star)_{ij}
=
\frac{G_{ij}}{1+\lambda E_{ij}^2}.
$$

这里 $G$ 是不加 ELLA 约束时的 unconstrained gradient/update proposal。于是 ELLA 的几何意义很清楚：历史能量越高的坐标，新更新越被压缩。干扰界写作：

$$
\langle \Delta W_t^\star,W_{\text{past}}\rangle
\le
\frac{\|G\|_F}{2\sqrt{\lambda}}
\|E^{-1}\odot W_{\text{past}}\|_F.
$$

### 训练流程与实验设置

base LLM 冻结，在 Attention Q/V 层使用 LoRA rank 8。每个任务训练当前 LoRA，并加入 $L_{\text{ELLA}}$；任务结束后把 $\Delta W_t$ 加到 $W_{\text{past}}$。主方法无 replay；部分 baseline 使用 2% replay。

模型覆盖 T5-Base 220M、T5-Large 770M、T5-XL 3B、LLaMA-3.1 8B；主表包括 T5-Large 和 LLaMA-3.1-8B。T5 用 V100 16GB，LLaMA 用 A40 48GB；Standard/Long Sequence 用 1 epoch，总 batch 32；TRACE 按任务训练 5、3、7、5、3、5、5、7 epochs。

数据包括 Standard CL 三种 order；Long Sequence 三种 15-task order；TRACE Order 7 为 c-stance、fomc、meetingbank、py150、scienceqa、numglue-cm、numglue-ds、20minuten；unseen GA 用 MMLU、GSM8K、BBH、AGIEval、PIQA。baselines 很多，包括 SeqFT、SeqLoRA、IncLoRA、SeqSVD、EWC、LwF、L2P、O-LoRA、O-LoRA+MIGU、DATA、DATA+Replay、LFPT5、SeqLoRAReplay、Recurrent-KIF。

### 关键结果与限制

| 设置 | ELLA | 对照 |
|---|---:|---|
| T5-Large Standard CL OA | `79.9` | O-LoRA `71.6`，DATA+Replay `75.9`，Recurrent-KIF `78.4` |
| T5-Large Long Sequence OA | `73.6` | O-LoRA `65.3`，DATA+Replay `74.3`，Recurrent-KIF `77.8` |
| T5-Large TRACE OA | `40.0` | O-LoRA `23.1`，DATA+Replay `36.5` |
| LLaMA3.1-8B | SC `77.57`，LS `74.18`，TRACE `33.29` | DATA+Replay SC `76.83`，LS `73.44`，TRACE `34.16` |
| unseen GA | DeltaGA `+5.48` | O-LoRA `-14.81`，SeqLoRAReplay `-16.81`，DATA `-110.14` |

效率表里，T5-Large 上 ELLA trainable params `0.062`、storage `4.19MB`、replay `0`、time/epoch `4.5` min；O-LoRA storage `31.46MB`，DATA storage `147.46MB` 且 replay `2%`。rank 消融显示 LoRA dim 8 最优：dim 2/4/8/16 平均为 `74.46/75.36/79.92/77.07`。

机制图也支持这个故事：$\lambda=0$ 时旧任务 batch 的 prediction loss spike 更多；ELLA 减少 opposing direction weight change；$\lambda$ 过小会遗忘、过大会挡住新任务，中间值最好。

限制是：没有评估 70B 级模型；数百任务扩展未知；多模态 CL 未覆盖；它不自动解决 base model 的偏见或有害输出。还有一个公平提醒：DATA+Replay、SeqLoRAReplay、Recurrent-KIF 等方法使用 replay 或更重结构，不能和 replay-free ELLA 直接按一行数字排名。

过渡到最后一问：如果我们不再相信“正交越硬越好”，那还有一个更基础的问题。LoRA 是 $\Delta W=BA$，只约束 $A$ 或某个子空间，优化器会不会从 $B$ 逃出去？

## 五、SFOR/BOD 作者稿：双线性 LoRA 会不会绕过正交？

这部分篇幅最长，因为它直接回答上一篇结尾的问题：O-LoRA 式正交保护到底哪里可能失效。这里讨论的是用户提供的作者稿；我只把它作为 author-provided draft 解读，不提供公共 URL，也不引用稿件里的模板性提交信息。

### 问题：只约束 routing matrix 够吗？

许多正交 LoRA 方法会约束 routing matrix $A$，希望 $A$ 避开旧任务特征方向。但 LoRA 更新不是线性的单矩阵参数，而是：

$$
\Delta W=BA,
$$

其中：

$$
B\in\mathbb{R}^{d_{\text{out}}\times r},
\quad
A\in\mathbb{R}^{r\times d_{\text{in}}}.
$$

作者稿的核心诊断是：如果 $A$ 被正交约束挡住，而 $B$ 没被约束，新任务学习压力会转移到 $B$。这会产生 **Bilinear Optimization Divergence (BOD)**：basis norm inflation、subspace distortion，以及最终遗忘。

这不是“正交约束没调好”的说法，而是更强的判断：在双线性参数化里，单边约束本身会改变另一个因子的优化动力学。

### Taylor 假设为什么会坏掉？

旧任务损失的 Taylor 展开需要先把矩阵更新向量化。令

$$
\delta w=\operatorname{vec}(\Delta W),
$$

则维度一致的写法是：

$$
\Delta L_{\text{old}}
\approx
\nabla_{w}L_{\text{old}}^\top\delta w
+\frac{1}{2}\delta w^\top H_{\text{old}}\delta w
+O(\|\delta w\|^3).
$$

传统正交论证常依赖两个假设：

1. 旧任务已接近局部最优，$\nabla_{\text{old}}\approx 0$。
2. 更新足够小，$\|\Delta W\|_F$ 让三阶项可以忽略。

但 LoRA 的真实增量是：

$$
\Delta W
=
(B+\Delta B)(A+\Delta A)-BA
\approx
\Delta B A+B\Delta A.
$$

当 $A$ 被投到旧任务特征的 null-space，KKT 条件可能让 $A$ 在约束边界停住；但全局梯度还有 residual $G_{\text{res}}$。此时 $B$ 的梯度为：

$$
\nabla_B L=G_{\text{res}}A^\top.
$$

因为 $B$ 没有被约束，如果残差方向稳定，就会出现非零期望偏置：

$$
\mathbb{E}[\nabla_B L]\neq 0.
$$

在 AdamW 下：

$$
m_t^{(B)}
=
\beta_1m_{t-1}^{(B)}+(1-\beta_1)\nabla_BL,
$$

更新近似为：

$$
\Delta B\approx
\eta\frac{m_t^{(B)}}{\sqrt{v_t^{(B)}+\epsilon}}.
$$

作者稿的 Theorem 2 形式可以概括为：若 $A$ 被 $P_{\text{null}}$ 严格约束而 $B$ 未约束，且 $\mu=\mathbb{E}[\nabla_B L]\neq 0$，AdamW 会把随机游走式的 $O(\eta\sqrt{T})$ 抬升为线性 norm inflation：

$$
\|B_T\|_F
\ge
\|B_0\|_F+\Omega(\eta T).
$$

一旦 $B$ 膨胀，$\|\Delta W\|_F$ 会离开二阶 Taylor 的局部可信尺度。若 Hessian 为 $\rho$-Lipschitz，三阶项相对二阶项可忽略需要更像

$$
\|\Delta W\|_F\ll \lambda_{\min}/\rho
$$

这样的局部启发式尺度，而不是一个硬边界。BOD 的意思就是：正交论证依赖的小更新假设，可能被双线性优化自己破坏。

### SFOR：冻结逃逸通道

SFOR 的修复很朴素，也很硬：

1. Task 1 同时训练 $A,B$，得到共享 basis。
2. 对 $k\ge 2$，冻结 $B$，即 $\Delta B_k=0$。
3. 后续只训练 $A$，并把 $A$ 的更新限制在旧任务 core features 的 null-space。

这样：

$$
\Delta W=B_{\text{frozen}}\Delta A.
$$

双线性系统退化成线性 routing 系统，$B$ 这条补偿通道被封住。

约束写作：

$$
AV_{\text{core}}=0,
$$

$P_{\text{null}}$ 是旧任务 core features $V_{\text{core}}$ 的 null-space projector。

### WRP：AdamW 会撕裂约束

作者稿还指出，光做 projected gradient 不够。AdamW 的实际物理更新不是“干净的投影梯度”，而是：

$$
A_t
\leftarrow
A_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t+\epsilon}}
-\eta\lambda A_{t-1}.
$$

这里有两个 tearing 通道：非均匀 variance scaling，以及 decoupled weight decay。

WRP 的做法是先让 AdamW 给出真实更新：

$$
\Delta A_{\text{actual}}=A_t-A_{t-1},
$$

再投影 residual：

$$
A_t^{\text{safe}}
=
A_{t-1}
+\Delta A_{\text{actual}}P_{\text{null}}.
$$

稿件也提到等价 weight projection $A_t^{\text{safe}}=A_t^{\text{raw}}P_{\text{null}}$，但实现采用 residual form 来降低数值漂移。这里必须把稿件内部的一处张力摊开：residual form 和 whole-weight projection 要等价，需要

$$
A_{t-1}P_{\text{null}}=A_{t-1}.
$$

在这个 invariant 下，若对 $A$ 做 decoupled weight decay，残差里的 decay 项投影后是

$$
(-\eta\lambda A_{t-1})P_{\text{null}}
=-\eta\lambda A_{t-1},
$$

并不会自动归零。可是稿件后面又用 $A_{\text{old}}P_{\text{null}}\approx 0$ 来说明 $A$ 侧 weight decay 会被 WRP 截断。两句话在同一个 $P_{\text{null}}$ 定义下不能同时成立：前者说 $A$ 已在 null-space 内，后者说 $A$ 与 null-space 近似正交。因此我不会把 “A 侧 weight-decay paradox” 当成已经严密成立的结论，只能把它记为作者稿里需要澄清的符号/定义问题。

$B$ 侧的直觉仍然有意义，但也应写成假设：即便 weight decay 压低 $\|B\|_F$，若 $\Delta B(A_{\text{old}}V_{\text{core}})$ 非零，就可能带来一阶历史干扰。也正因为这个变量没被干净隔离，实验严格设置 `weight_decay=0` 更像是在排除 weight decay，把 tearing 归因集中到 AdamW 的 momentum/variance scaling。

### Mature SFOR：过稳之后，塑性从哪里来？

基础 SFOR 的问题也明显：如果 Task 1 的 basis 太窄，后续异质任务会学不动。作者稿把这叫 plasticity bottleneck，于是引入 Mature SFOR。

第一步是 adaptive subspace budgeting。对第 $l$ 层 feature covariance 做 SVD，奇异值为 $\{\sigma_i\}$，动态 rank：

$$
k_{\text{var}}^{(l)}
=
\min\left\{k:
\frac{\sum_{i=1}^{k}\sigma_i^2}
{\sum_{i=1}^{r}\sigma_i^2}
\ge
\tau_{\text{var}}
\right\}.
$$

再 clamp：

$$
k^{(l)}
=
\max(k_{\min},\min(k_{\text{var}}^{(l)},k_{\max})).
$$

$\tau_{\text{var}}$ 控制保留强度，$k_{\max}$ 保证仍给新任务留下至少 $d_{\text{in}}-k_{\max}$ 维 null-space。

第二步是 Safe Orthogonal Basis Expansion (SOBE)。给旧 basis：

$$
B_{\text{old}}\in\mathbb{R}^{d_{\text{out}}\times r_{\text{old}}},
$$

随机扩展：

$$
\Delta B_{\text{raw}}\in\mathbb{R}^{d_{\text{out}}\times \Delta r}.
$$

用 Gram-Schmidt 得到正交新增 basis：

$$
\Delta B_{\perp}
=
\Delta B_{\text{raw}}
-B_{\text{old}}(B_{\text{old}}^\top B_{\text{old}})^{-1}
B_{\text{old}}^\top\Delta B_{\text{raw}}.
$$

扩展为：

$$
B_{\text{new}}=[B_{\text{old}},\Delta B_{\perp}],
\quad
A_{\text{new}}=
\begin{bmatrix}
A_{\text{old}}\\
0_{\Delta r\times d_{\text{in}}}
\end{bmatrix}.
$$

新增 $A$ 行初始化为 0，所以扩展瞬间 forward zero-interference。反传 hook 写作：

$$
\nabla_{B_{\text{new}}}L
=
[0_{d_{\text{out}}\times r_{\text{old}}},
\nabla_{\Delta B_{\perp}}L],
$$

只校准新 basis。配置上，LLaMA-2-7B 使用 $\{max\_k=20,\Delta r=1,\tau_{\text{var}}=0.93\}$；Qwen-2.5-3B 使用 $\{max\_k=32,\Delta r=2,\tau_{\text{var}}=0.95\}$。

### 实验设置

作者稿评估标准 5-task classification、TRACE 异质 instruction tuning、15-task long-horizon。模型为 LLaMA-2-7B 和 Qwen-2.5-3B，另有 1.5B/32B 微观探针。base SFOR 固定 LoRA rank $r=8$，零参数增长；Mature SFOR 有界扩展，例如 rank `8 -> 13`。优化器为 AdamW，$\beta_1=0.9,\beta_2=0.999,lr=10^{-4},weight\_decay=0$，有效 batch size 8。

classification 任务为 AG News、Amazon、Yelp、DBpedia、Yahoo，并测了三个 sequence permutation。TRACE 主表任务为 GSM8K、ScienceQA、DialogSum、CodeAlpaca、AG News。baselines 包括 SeqFT、EWC、Base SVD、O-LoRA；附录还比较 SD-LoRA。指标包括 Avg Acc、Avg FM、BWT、Drift Sim、最大 $\|B\|_F$、projection/orthogonal error。

### 关键结果：不要藏起反例

这篇稿件最需要透明呈现的地方是：**在同质 7B classification headline 上，O-LoRA 明显强于基础 SFOR**。

| 设置 | SFOR | O-LoRA | 我会怎么写 |
|---|---:|---:|---|
| LLaMA-2-7B 1-epoch classification | Avg Acc/FM `90.52/6.61` | `97.54/2.00` | O-LoRA 表面指标更好 |
| LLaMA-2-7B 3-epochs classification | `90.97/8.89` | `98.60/0.82` | 深训同样 O-LoRA 更高 |
| Qwen-2.5-3B 1-epoch classification | `74.05/15.25` | `71.29/31.80` | 小模型里 SFOR 遗忘更低 |
| Qwen-2.5-3B 3-epochs classification | `74.26/21.49` | `59.27/47.64` | 训练更深时 O-LoRA 崩得更明显 |

作者稿把这解释为 **Scaling Buffer Effect**：7B 的冗余容量会暂时吸收不完美约束带来的不一致，让 O-LoRA 看起来非常稳；小模型或深训时，这个缓冲变弱，BOD 暴露出来。

TRACE 上，Mature SFOR 的结果更能体现它的目标：

| TRACE / LLaMA-2-7B 1-epoch | GSM8K | ScienceQA | DialogSum | CodeAlpaca | AG News | Avg Acc | BWT | Drift Sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SeqFT | 4.00 | 60.40 | 32.39 | 22.00 | 94.00 | 42.56 | -1.59 | 0.9996 |
| O-LoRA | 3.60 | 58.80 | 16.19 | 16.00 | 93.20 | 37.56 | -7.74 | 0.9996 |
| Semi-Frozen | 14.80 | 47.60 | 18.22 | 13.20 | 81.60 | 35.08 | +0.40 | 0.9999 |
| Mature SFOR | 5.20 | 69.60 | 37.25 | 16.80 | 90.40 | 43.85 | +5.37 | 0.9998 |

Qwen-2.5-3B TRACE 1-epoch 中，Mature SFOR 为 `58.31%/+0.31/0.7617`，O-LoRA 为 `55.67%/-5.51/0.6295`，Base SVD 为 `56.71%/-4.62/0.5661`。但 3-epoch 的 Qwen TRACE 里，Semi-Frozen 是 `59.91%/-1.10/0.8811`，Mature 是 `58.47%/-1.81/0.6846`，EWC 是 `59.99%/-1.71/0.8457`。所以我不会照抄“最高 AA”；更准确的说法是：SFOR 接近并列级别，且 BWT/结构稳定性更符合作者想证明的机制主张。

机制消融也很重要。Table 12 的 constructive ablation 显示：

| 变体 | $\|B\|_F$ | projection error | Acc/FM |
|---|---|---|---|
| Base SVD | `~0.386 exploding` | `~1e-3` | `85.28/17.23` |
| + Frozen B | `0.134 locked` | `~1e-4` | `89.76/6.41` |
| + WRP | `0.157 locked` | `<1e-6` | `90.83/7.89` |
| + Scheduler | `0.157 locked` | `<1e-6` | `90.53/6.59` |

这说明冻结 $B$ 解决 norm inflation，WRP 解决 AdamW tearing。Figure 2 的 norm dynamics/orthogonal violation 也支持这个说法：baselines 的 $\|B\|_F$ 膨胀，3B 中 soft constraints collapse，hard projection 仍受 AdamW tearing；SFOR+WRP 把 $\|B\|_F$ 锁住，并把 orthogonal error 压到数值精度。

长 horizon 和微观探针提供了另一层证据：15-task 中 Base SVD 在 Task 1/2 accuracy 上出现 cyclic interference，SFOR 更稳定；1.5B/32B 探针说明大模型能缓冲但不能消除 BOD；提高 LoRA rank 反而可能给 $B$ 更大的补偿通道。

### 假设与限制

SFOR 的 replay 条件很干净：不 replay。base SFOR 严格零参数增长，rank `8 -> 8`；Mature SFOR 允许有界扩展，如 `8 -> 13`。推理不应依赖 task ID，但训练需要任务边界，并且 Task 1 是 basis anchor。

限制也必须写清楚：

- 同质 7B classification 上，O-LoRA headline 更强；SFOR 的主张不是“所有表格都赢”，而是“BOD 是真实失稳机制，且在小模型、深训、异质任务和微观探针中显现”。
- base SFOR 在异质 TRACE 上会过稳，Mature SFOR 是为释放塑性引入的受控放松。
- null-space 在极长 horizon 会饱和。
- 实验主要围绕 LLaMA-2-7B、Qwen-2.5-3B、rank 8；部分表格是 single-seed representative，虽然稿件称稳定指标方差可忽略。
- 有些抽取表行不完整，我没有引用缺失行。

## 六、横向比较：哪些数字能比，哪些不能比？

| 维度 | Sculpting Subspaces | OA-Adapter | OLieRA | ELLA | SFOR/BOD |
|---|---|---|---|---|---|
| 更新对象 | 全参数 projected fine-tuning | Adapter modules | 每任务 LoRA + 乘性更新 | LoRA + 聚合能量正则 | 冻结 basis 的 LoRA routing + WRP |
| 主要问题 | 全参表达力与少遗忘 | 动态预算与正交 | 参数几何与正交 | 选择性去相关 | 双线性优化逃逸 |
| replay | 不参与梯度 replay；任务边界需上一任务样本或缓存统计量 | 主方法无 | 无 | 主方法无 | 无 |
| 参数增长 | 固定 | task-specific adapter 管理，预算更省 | 每任务 LoRA 增长 | 无架构扩展，维护聚合矩阵 | base 零增长，Mature 有界增长 |
| 推理 task ID | 不依赖 | 声称不需显式 task ID | 继承 O-LoRA | 不依赖 | 不依赖 |
| 代表模型 | T5-Large / LLaMA-2 7B | T5-large / Llama-7B | T5-large / LLaMA-7B | T5-Large / LLaMA3.1-8B | LLaMA-2-7B / Qwen-2.5-3B |
| 标准分类 headline | T5 `75.9` | T5 `76.0` | T5 `79.6` | T5 `79.9` | 7B SFOR `90.52`，但 O-LoRA `97.54` |
| 异质任务 | TRACE `48.4` | SuperNI `29.3` | 未报 TRACE | TRACE T5 `40.0`，LLaMA `33.29` | TRACE Mature 7B `43.85/+5.37` |

为了避免“平均值”遮住任务顺序，这里把主要 task-order 覆盖压缩列一下：

```text
OA / OLieRA / ELLA standard orders:
1. DBpedia -> Amazon -> Yahoo -> AG
2. DBpedia -> Amazon -> AG -> Yahoo
3. Yahoo -> Amazon -> AG -> DBpedia

Long-order representative used by OA/OLieRA/ELLA appendices:
MNLI -> CB -> WiC -> COPA -> QQP -> BoolQA -> RTE -> IMDB
-> Yelp -> Amazon -> SST-2 -> DBpedia -> AG -> MultiRC -> Yahoo
The remaining two long-order permutations are listed in the corresponding paper appendices.

ELLA TRACE order:
c-stance -> fomc -> meetingbank -> py150 -> scienceqa
-> numglue-cm -> numglue-ds -> 20minuten

SFOR 5-task permutations:
1. AG News -> Amazon -> Yelp -> DBpedia -> Yahoo
2. Amazon -> Yelp -> DBpedia -> Yahoo -> AG News
3. Yahoo -> DBpedia -> Yelp -> Amazon -> AG News
```

具体警告：

- Standard CL 的“5-task”并不总是五个任务 order；有些表实际使用 DBpedia/Amazon/Yahoo/AG 四个任务顺序。
- TRACE、SuperNI、Long Sequence 不是同一个 benchmark，不能横比绝对值。
- T5-Large、T5-XL、LLaMA-2-7B、LLaMA-3.1-8B、Qwen-2.5-3B 的能力和 prompt format 差异很大。
- ProgPrompt、PerTaskFT、MTL 使用 task prompt、独立模型或联合数据等更强约束，不应和 replay-free single-model 方法直接排名。
- AA、OA、Avg Acc、FWT、BWT、FM、Drift Sim 不是同一个指标。
- SFOR 的 scaling-buffer 结论不等于“7B 上 SFOR 永远最好”；恰恰相反，同质 7B classification 中 O-LoRA headline 更高。

## 七、旁注：C-LoRA、DOC、SLICE 为什么不进五篇主线？

这三条线都相关，但我没有把它们做成深度 profile。

| 方法 | 为什么相关 | 为什么这里只做旁注 |
|---|---|---|
| C-LoRA | 用 learnable routing matrix $R$ 把 LoRA 写成 $\Delta W=ARB$，也关心子空间复用与正交 | 本地材料显示其实验主体是 class-incremental vision/ViT，数据集为 Split CIFAR-100、Split ImageNet-A、Split CUB-200、Split CAR196，不是这篇的 LLM 主线 |
| DOC | Dynamic Orthogonal Continual fine-tuning 跟踪 functional directions 的漂移，用梯度和 LoRA increments 通过 online PCA 更新历史功能方向，并把新更新切到这些方向的正交补 | 它是 LLM continual fine-tuning 路线，实验覆盖 LLaMA-7B/13B 和 T5-Large，数据来自 CL benchmark、GLUE、SuperGLUE 与 IMDB；但它的重心是动态功能方向跟踪，不是 LoRA 双线性 BOD |
| SLICE | 用 gradient surgery 调和当前任务梯度和历史任务梯度，再通过 truncated SVD 初始化 LoRA adapter；评估 SuperNI、TRACE 和梯度冲突更强的 NI-Seq-Opposite | 它使用 Llama-3.2-3B-Instruct、rs-LoRA rank 64 主设定，并且只在初始化时用 prior-task sample/replay buffer 估计梯度后丢弃；更像 LoRA 初始化/梯度冲突路线，不是运行时正交约束的核心链条 |

它们提醒我一件事：正交并不是唯一语言。有人从 routing 来看，有人从功能方向漂移来看，有人从梯度冲突初始化来看。本文只沿“正交子空间到双线性优化失稳”这条线走到底。

## 八、我现在的理解

如果把五篇连起来看，我会这样理解这条发展线：

Sculpting Subspaces 先把问题从 LoRA 拉回全参数空间：旧知识可能藏在高奇异值/高曲率方向，新任务可以走正交补。

OA-Adapter 说：就算我们在小模块里做正交，也不能假设每层、每个任务都需要同样预算。

OLieRA 说：即便子空间正交，LoRA 的加性更新也许不是最自然的参数几何，乘性更新能更温和地保留结构。

ELLA 说：严格正交会消耗自由度，也可能阻断迁移；与其禁止所有 overlap，不如选择性压低高能历史方向。

SFOR/BOD 则把刀口插进 LoRA 参数化本身：$\Delta W=BA$ 是双线性的。只约束 $A$，并不等于约束真实更新；优化器可以通过 $B$ 的 norm inflation 绕开正交。SFOR 的强硬做法是冻结 $B$，WRP 再修正 AdamW 的物理更新；Mature SFOR 则承认异质任务需要受控扩展。

我现在最不愿意说的一句话是：“正交 LoRA 可以防止遗忘。”这句话太短，短到把所有关键假设都藏起来了。

更准确的说法也许是：

> 正交约束能减少某些可测子空间里的干扰，但它是否真正保护知识，取决于更新对象、参数化方式、优化器物理轨迹、任务异质性、模型规模，以及未来任务还能使用多少剩余自由度。

## 九、开放问题

1. 旧任务“知识”到底应该用权重奇异方向、梯度方向、激活功能方向，还是 LoRA 更新方向来表示？
2. 正交保护和正迁移之间有没有可学习的门控，而不是固定的硬约束或单一 $\lambda$？
3. BOD 是否会在 QLoRA、DoRA、AdaLoRA、Mixture-of-LoRA 等变体中以不同形式出现？
4. Mature SFOR 的 basis anchor 对首任务语义覆盖很敏感，能不能用预训练数据或无监督探针初始化更通用的 basis？
5. 如果任务边界不可见，正交子空间方法如何在线发现“该扩展还是该复用”？
6. 长到几百个任务时，null-space 饱和会怎样表现？是缓慢退化，还是突然断崖？
7. 评价是否应从 AA/BWT 扩展到机制指标，比如 $\|B\|_F$、orthogonal error、Drift Sim、feature covariance rank？
8. 大模型的 Scaling Buffer 是好事还是坏事？它让系统短期更稳，但也可能掩盖机制问题。

我的个人结论是：正交不是答案本身，而是一种坐标系。它帮我们看见干扰发生在哪里，也会让我们误以为“看不见干扰”就是“没有干扰”。真正值得追的，是正交约束、低秩参数化和优化器动力学三者之间的缝隙。BOD 这个问题之所以有意思，正是因为它把这个缝隙从经验现象变成了一个可以被公式、探针和消融共同检查的对象。

## 参考

- Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning, arXiv:2504.07097, https://arxiv.org/abs/2504.07097
- Adaptive Budget Allocation for Orthogonal-Subspace Adapter Tuning in LLMs Continual Learning, arXiv:2505.22358, https://arxiv.org/abs/2505.22358
- Orthogonal Low-rank Adaptation in Lie Groups for Continual Learning of Large Language Models, arXiv:2509.06100, https://arxiv.org/abs/2509.06100
- ELLA: Efficient Lifelong Learning for Adapters in Large Language Models, arXiv:2601.02232, https://arxiv.org/abs/2601.02232
- On the Instability of Orthogonal LoRA in Continual Learning: Bilinear Optimization Divergence, Scaling Buffer Effects, and Semi-Frozen Routing, author-provided draft.
- C-LoRA: Continual Low-Rank Adaptation for Pre-trained Models, arXiv:2502.17920, https://arxiv.org/abs/2502.17920
- Dynamic Orthogonal Continual Fine-tuning for Mitigating Catastrophic Forgettings, arXiv:2509.23893, https://arxiv.org/abs/2509.23893
- Low-Rank Adapters Initialization via Gradient Surgery for Continual Learning, arXiv:2605.12752, https://arxiv.org/abs/2605.12752

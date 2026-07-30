---
title: "个人总结正交类LoRA发展"
summary: "顺着多篇近期工作，理解正交 LoRA 持续学习从子空间隔离、参数不碰撞、梯度投影、合并式更新到双线性失稳的关键假设。"
date: 2026-07-30
tags: ["大模型","持续学习","LoRA","正交子空间"]
category: "论文解读"
draft: false
---

## 一切的开始：O-LoRA

论文：[Orthogonal Subspace Learning for Language Model Continual Learning](https://arxiv.org/abs/2310.14152)，2023.10  
官方代码仓库：[https://github.com/cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA)

【注：虽然 O-LoRA 在论文中写的是 \(h=(W+AB)x\)，但是他们代码仓库实际上是 \(h=(W+BA)x\)，且我本人更习惯写为 \(\text{LoRA}=BA\)，所以就沿用 \(BA\) 写法。】

$$
h = W_{\text{init}}x + BAx
$$

大家都知道，LoRA 的先作用矩阵，也就是本文中的 \(A\) 矩阵，是**决定 LoRA 分支会对哪些输入特征方向产生响应的关键**。

先来说为什么：

**因为 \(A\) 矩阵是先和 \(x\) 相乘的矩阵。**

把 \(A\) 拆成一行一行看：

$$
A=
\begin{bmatrix}
a_1^\top\\
a_2^\top\\
\vdots\\ 
a_r^\top
\end{bmatrix}
$$

那么：

$$
z=Ax
$$

等价于：

$$
\begin{aligned}
z_1 &= a_1^\top x,\\
z_2 &= a_2^\top x,\\
&\vdots\\
z_r &= a_r^\top x.
\end{aligned}
$$

也就是说，\(A\) 的每一行都在问：

当前 hidden state \(x\) 里，有没有我关心的这个方向？

然后 \(B\) 再把这些低维响应写回输出空间：

$$
Bz = z_1b_1+z_2b_2+\cdots+z_rb_r
$$

所以：

- \(A\)：决定“看哪些方向 / 由哪些输入特征触发”；
- \(B\)：决定“触发后往输出空间怎么写”。

因此如果连续学习里想保护旧任务，一个很自然的想法是：**让新任务的 \(A_{\text{new}}\) 不要看旧任务依赖的特征方向**。

所以就有了 O-LoRA 的设计。

【先整体说下 O-LoRA，他们的设计总览就是每来一个新任务，就冻结上一个任务训练的 LoRA 矩阵，并增添一个新的 LoRA 矩阵，通过正则方法确保新 LoRA 矩阵训练过程中尽可能和旧 LoRA 矩阵保持正交。】

按本文的 \(BA\) 记法，一个自然写法是让新旧任务的输入侧矩阵满足：

$$
A_tA_i^\top \approx 0,\quad i<t
$$

对应的正交正则可以写成：

$$
L_{\text{orth}}(A_t)
=
\sum_{i<t}\left\|A_tA_i^\top\right\|_F^2
$$

换句话说就是，新的 LoRA 矩阵的 \(A\) 矩阵会和每一个旧任务 LoRA 的 \(A\) 矩阵做内积。

最理想的情景当然是：

$$
A_tA_i^\top = 0
$$

这样新任务和旧任务在输入侧响应方向上尽可能分开。这里我会稍微加一句谨慎的话：这保证的是 \(A\) 子空间正交，**不严格等价于完整参数更新 \(\Delta W=BA\) 一定完全不干扰**。后面 On the Instability 这篇论文其实就是在追这个问题。

## CV 领域类似的探索：Orthogonal Adaptation

论文：[Orthogonal Adaptation](https://arxiv.org/abs/2312.02432)，2023.12

背景就是图片生成领域目前存在一个问题：**如果我训练了几个单概念的 LoRA，比如猫 LoRA，狗 LoRA，我现在想生成一张同时有猫和狗的图片，就会出现猫和狗概念混杂的问题**。

这个问题被称为串扰，crosstalk。

现有几种方法各有缺陷：

1. 联合训练：问题是如果有 \(n\) 个物体概念，你想得到任何两者同框的图片都需要训练一次，训练次数会需要 \(2^n\)，更别提还有 \(n\) 个概念混杂的场景。
2. 直接相加 LoRA，会出现 crosstalk 问题。
3. Mix-of-Show 的 Gradient Fusion，但是合并三个概念就需要 15 分钟，消耗太大且没有泛化性，和联合训练一个问题。

那就尝试解决 crosstalk，因为这样的话只需要训练各自的 LoRA，然后可以在 1s 内相加 LoRA 并生成联合概念图片。

作者的答案是：**冻结 \(\Delta W=BA\) 中的 \(A\)，只训练 \(B\)，而每个任务，或者说概念，分配到的 \(A\) 矩阵都是事先分配的。**

先说为什么。

crosstalk 被认为是两个 LoRA 矩阵在相同参数上的互相干扰。

对于概念 \(i\)，单独训练后的某层输出是：

$$
O_i(X_i)=(W+\Delta W_i)X_i
$$

其中 \(X_i\) 是概念 \(i\) 在该层的输入特征。把概念 \(j\) 合并进来后：

$$
\begin{aligned}
\hat O_i(X_i)
&=
(W+\Delta W_i+\Delta W_j)X_i\\
&=
O_i(X_i)+\Delta W_jX_i
\end{aligned}
$$

最后一项：

$$
\Delta W_jX_i
$$

就是概念 \(j\) 对概念 \(i\) 造成的串扰。理想情况是它等于零，但 \(X_i\) 通常覆盖很大的输入空间，强行要求它处处为零，会把 \(\Delta W_j\) 限制得几乎不能学习。

因此作者把要求放松为：只在输入的主要投影子空间中消除串扰。

其实是和 O-LoRA 同样的思路，因为 \(A\) 矩阵是**决定 LoRA 分支会对哪些输入特征方向产生响应的关键**，所以只需要保证新旧任务在 \(A\) 矩阵上正交即可。

本论文用的方法和 O-LoRA 还是有区别的，他们没有使用正则方法，而是先搞了一个正交基矩阵 \(O\)，满足：

$$
OO^\top=I
$$

这里有个我之前也容易下意识说错的点：\(O\) 不等于一定是单位矩阵 \(I\)。单位矩阵只是最简单的正交矩阵；任意旋转后的正交基也可以满足 \(OO^\top=I\)。

然后从这个共享正交基 \(O\) 里给不同概念抽取不重合的 \(r\) 个方向作为 \(A_i\)。只要新旧任务抽到的方向不重合，就可以保证：

$$
A_iA_j^\top=0
$$

这就比 O-LoRA 的软正则更硬：O-LoRA 是训练时“尽量正交”，Orthogonal Adaptation 是初始化时就把 \(A\) 分配到互相正交的方向上。

## InfLoRA 和 KeepLoRA：从 O-LoRA 上改进

论文：

- [InfLoRA](https://arxiv.org/abs/2404.00228)，2024.4
- [KeepLoRA](https://arxiv.org/abs/2601.19659)，2026.1

先说 InfLoRA。他们是**冻结 \(A\) 矩阵，训练 \(B\) 矩阵，\(A\) 矩阵是通过算法算出来的**。说到这是不是和 Orthogonal Adaptation 这篇论文有点像？只不过 \(A\) 矩阵找法不一样。

设计依据是当前任务在该层的输入特征矩阵：

$$
H_t=[h_1^t,\dots,h_n^t]
$$

因为线性层：

$$
e=Wh
$$

其梯度满足：

$$
\frac{\partial L}{\partial W}
=
\frac{\partial L}{\partial e}h^\top
$$

所以梯度方向和输入 \(h\) 的 span 有关。

核心就是一个思想：**\(H_t\) 近似当前任务梯度空间**。

然后 InfLoRA 会把当前任务输入特征里落在旧任务空间 \(M_t\) 里的部分减掉：

$$
\hat H_t = H_t-M_tM_t^\top H_t
$$

这里 \(M_tM_t^\top H_t\) 就是 \(H_t\) 投影到旧任务空间里的部分。减掉它之后，\(\hat H_t\) 就更像“当前任务还可以安全使用的方向”。之后再对 \(\hat H_t\) 做 SVD，拿主方向来构造固定的 \(A_t\)。

再说 KeepLoRA。KeepLoRA 认为 InfLoRA 只考虑了“不要忘记旧任务”，却没有显式保护预训练模型原本的通用知识。

它先构造两个需要保护的空间。

第一，对预训练权重做 SVD，取大奇异值对应的主空间 \(W_p\)。作者认为这里主要保存预训练通用知识。

第二，用 \(M_{t-1}\) 保存旧任务输入特征的主要方向。

然后计算当前任务第一步梯度 \(G_t\)，删除落在这两个受保护空间中的部分：

$$
\hat G_t
=
G_t
-W_pW_p^\top G_t
-M_{t-1}M_{t-1}^\top G_t
$$

对 \(\hat G_t\) 做 SVD，取前 \(r\) 个方向初始化固定矩阵 \(A_t\)，之后只训练 \(B_t\)。

**笔者认为这篇文章只不过是在 InfLoRA 的基础上多保护了预训练的语义空间，谈不上什么特别颠覆的创新【唯一创新的一点是参数 \(W\) 的大奇异值近似预训练主子空间？】。他们能中 ICLR 2026 我还是有点惊讶的，感觉有工作量的功劳。**

## Sculpting Subspaces：从 LoRA 到全参微调

论文：[Sculpting Subspaces](https://arxiv.org/abs/2504.07097)，2025.4

它的核心假设是：

**大奇异值方向保存重要的已有知识；小奇异值方向冗余度较高，可以用来学习新任务。**

对每层权重做：

$$
W=U\Sigma V^\top
$$

然后保留大奇异值对应的 \(U_{\text{high}},V_{\text{high}}\)，并从新任务梯度中删除对应部分：

$$
G_{\text{proj}}
=
G
-U_{\text{high}}U_{\text{high}}^\top
G
V_{\text{high}}V_{\text{high}}^\top
$$

它还通过层输入与输出的余弦相似度估计层重要性，为重要层保留更多大奇异方向。

同样是做 SVD 找大奇异值，InfLoRA 做法和本文还是有区别的。前者面向数据，是对从数据来的激活值近似主子空间；后者面向参数，是直接对模型权重的奇异方向做保护。

## On the Instability of Orthogonal LoRA in Continual Learning：指出 LoRA 正交类持续学习问题

这篇的核心批评是：正交 LoRA 持续学习如果把安全性主要压在输入侧/路由侧的 \(A\) 矩阵上，比如 O-LoRA、Orthogonal Adaptation、InfLoRA、KeepLoRA 这类路线，那么仍然可能有一个问题：

真实更新是：

$$
\Delta W=BA
$$

如果 \(B\) 仍然能训练，它就可能出现补偿性更新，导致 \(B\) 矩阵范数增大。范数增大未必永远是坏事，但是通过泰勒展开的分析可知，如果 \(\|\Delta W\|\) 变大，旧任务 loss 的高阶项就不能再被忽略，原来“局部小更新 + 正交保护”的理论近似会变得不可靠。

这个分析和 Sculpting Subspaces 里对旧任务损失做二阶近似的思路有点相通，都是在问：新任务更新到底有没有真的小到可以忽略高阶影响？

WRP，也就是 Weight Residual Projection，则是在处理另一个问题：AdamW 优化器可能会扭曲实际更新方向。哪怕梯度是正交的，实际参数更新中也会因为动量、二阶方差缩放、weight decay 等因素变得不再正交。

所以 WRP 不只投影梯度，而是投影真实的权重残差：

$$
A_t^{\text{safe}}
=
A_{t-1}
+
\bigl(A_t^{\text{raw}}-A_{t-1}\bigr)P_{\text{null}}
$$

这里的重点是：先让优化器产生真实更新 \(A_t^{\text{raw}}-A_{t-1}\)，再把这个真实残差投影回安全的 null-space。

当然，**只有在 \(B\) 被冻住之后，只限制 \(A\) 的实际更新方向才比较可靠**，原因也和前面一样：如果 \(B\) 还在动，那么真实的 \(\Delta W=BA\) 仍然可能从 \(B\) 这条路绕出去。

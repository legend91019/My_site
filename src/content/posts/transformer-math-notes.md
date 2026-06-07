---
title: "Transformer 架构中的数学美学与注意力机制推导"
summary: "系统推导 Transformer 中 Scaled Dot-Product Attention 缩放因子的数学原理，说明为什么要除以根号 d_k。"
date: 2026-05-18
tags: ["Transformer", "注意力机制", "深度学习数学", "表示学习"]
category: "深度学习"
draft: false
---

在自然语言处理和计算机视觉中，**Transformer** 已经成为主流大模型的基础组件。它最核心的算子，就是自注意力机制：

$$
\mathrm{Attention}(Q, K, V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

真正值得细究的问题是：为什么这里一定要除以 $\sqrt{d_k}$，而不是别的量？

---

## 自注意力的定义

先给出标准定义。

> **定义 1：缩放点积注意力（Scaled Dot-Product Attention）**
>
> 给定 Query 矩阵 $Q \in \mathbb{R}^{n \times d_k}$、Key 矩阵 $K \in \mathbb{R}^{m \times d_k}$、Value 矩阵 $V \in \mathbb{R}^{m \times d_v}$，
> 其输出定义为
>
> $$
> \mathrm{Attention}(Q, K, V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
> $$

其中：

- $n$ 和 $m$ 分别表示 Query 序列与 Key/Value 序列长度。
- $d_k$ 表示 Query 与 Key 的通道维度。
- $\sqrt{d_k}$ 就是所谓的缩放因子。

---

## 为什么是 $\sqrt{d_k}$

如果不做缩放，注意力分数会随着维度变大而迅速膨胀。

> **定理 1：高维随机向量点积的方差与维度成正比**
>
> 设 $q,k \in \mathbb{R}^{d_k}$ 为相互独立的随机向量，并假设每个分量都满足
>
> $$
> \mathbb{E}[q_i]=\mathbb{E}[k_i]=0,\qquad
> \mathrm{Var}(q_i)=\mathrm{Var}(k_i)=1
> $$
>
> 则它们的点积
>
> $$
> q \cdot k = \sum_{i=1}^{d_k} q_i k_i
> $$
>
> 满足
>
> $$
> \mathbb{E}[q \cdot k]=0,\qquad
> \mathrm{Var}(q \cdot k)=d_k
> $$

证明并不复杂。

首先，由独立性可得

$$
\mathbb{E}[q_i k_i]=\mathbb{E}[q_i]\mathbb{E}[k_i]=0
$$

所以

$$
\mathbb{E}[q \cdot k]
=
\mathbb{E}\left[\sum_{i=1}^{d_k} q_i k_i\right]
=
\sum_{i=1}^{d_k}\mathbb{E}[q_i k_i]
=0
$$

再看单项的方差：

$$
\mathrm{Var}(q_i k_i)
=
\mathbb{E}[(q_i k_i)^2]-(\mathbb{E}[q_i k_i])^2
$$

由于 $q_i$ 与 $k_i$ 独立，且都满足单位方差，

$$
\mathrm{Var}(q_i k_i)
=
\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]
=1
$$

于是总方差为

$$
\mathrm{Var}(q \cdot k)
=
\mathrm{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right)
=
\sum_{i=1}^{d_k}\mathrm{Var}(q_i k_i)
=
d_k
$$

> **说明**
>
> 这意味着维度越高，未缩放点积的波动越大。比如当 $d_k=1024$ 时，注意力分数已经可能进入一个非常尖锐的范围。

---

## 为什么这会伤害 Softmax

注意力最终还要经过 Softmax：

$$
s_i=\mathrm{softmax}(z)_i=\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

如果输入向量 $z$ 的不同分量差异很大，那么 Softmax 会迅速接近 one-hot：

$$
s_m \approx 1,\qquad s_j \approx 0 \quad (j \neq m)
$$

这会带来两个后果：

1. 梯度接近饱和区，训练容易不稳定。
2. 注意力会过于集中，难以在多个相关位置之间平滑分配权重。

因此，我们需要把点积的方差重新拉回常数量级。对 $q \cdot k$ 除以 $\sqrt{d_k}$ 后，

$$
\mathrm{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right)
=
\frac{1}{d_k}\mathrm{Var}(q \cdot k)
=
\frac{d_k}{d_k}
=1
$$

这就是缩放因子取 $\sqrt{d_k}$ 的核心原因。它并不是经验魔法，而是一个直接来自方差控制的数学结论。

---

## 一个简短的数值验证

下面用 PyTorch 做个简单实验：

```python
import torch

dk = 1024
batch_size = 10000

q = torch.randn(batch_size, dk)
k = torch.randn(batch_size, dk)

dot_product = (q * k).sum(dim=-1)
print("unscaled variance:", dot_product.var().item())

scaled_dot_product = dot_product / (dk ** 0.5)
print("scaled variance:", scaled_dot_product.var().item())
```

理论上，第一项应接近 $1024$，第二项应接近 $1$。

---

## 小结

如果把 Transformer 看成一套优雅的数值系统，那么 $\sqrt{d_k}$ 的作用就是把注意力分数重新归一到可训练的尺度上。没有这一步，Softmax 很快会进入过度尖锐的区域；有了这一步，模型才能在高维空间里保持稳定的梯度流动。

后面如果继续写这个系列，我很想再把 **RoPE**、**LayerNorm** 和 **FlashAttention** 的数学直觉也整理出来。

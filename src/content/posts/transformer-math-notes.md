---
title: "Transformer 架构中的数学美学与注意力机制推导"
summary: "系统推导 Transformer 中 Scaled Dot-Product Attention 缩放因子的数学原理，深入证明除以根号 dk 如何稳定 Softmax 的方差，解决梯度消失问题。"
date: 2026-05-18
tags: ["Transformer", "注意力机制", "深度学习数学", "表示学习"]
category: "深度学习"
draft: false
---

在自然语言处理和计算机视觉中，**Transformer** 已经成为了几乎所有主流大模型（LLM）的基石。其核心算子 —— **自注意力机制 (Self-Attention)** 巧妙地通过向量内积捕捉了序列内任意两点之间的长距离关联。

本篇文章我们将以极其严谨的数学视角，系统推导其缩放因子 $\sqrt{d_k}$ 的由来，并探讨为什么在高维空间下，内积的方差膨胀会成为模型训练的“毒药”。

---

## 自注意力机制的定义

首先，我们给出注意力机制的代数描述：

<div class="definition">
<strong>定义 1：缩放点积注意力 (Scaled Dot-Product Attention)</strong><br/>
给定输入 Query 矩阵 $Q \in \mathbb{R}^{n \times d_k}$，Key 矩阵 $K \in \mathbb{R}^{m \times d_k}$，以及 Value 矩阵 $V \in \mathbb{R}^{m \times d_v}$。

其缩放点积注意力输出定义为：
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$
</div>

其中：
- $n$ 和 $m$ 分别代表 Query 序列与 Key/Value 序列的长度。
- $d_k$ 代表 Query 和 Key 的维度（通道数）。
- $\sqrt{d_k}$ 即为**缩放因子 (Scale Factor)**。

---

## 为什么缩放因子是 $\sqrt{d_k}$？

在 Vaswani 等人的原始论文《Attention Is All You Need》中提到，当 $d_k$ 较小时，点积注意力的表现与相加注意力（Additive Attention）相似；然而**当 $d_k$ 较大时，如果不进行缩放，点积注意力的表现会急剧变差**。

我们将这一现象提炼为以下数学定理并予以严格证明：

<div class="theorem">
<strong>定理 1：高维向量内积的方差膨胀性质</strong><br/>
设 Query 向量 $q \in \mathbb{R}^{d_k}$ 与 Key 向量 $k \in \mathbb{R}^{d_k}$ 为相互独立的随机向量。假设它们的每个分量 $q_i$ 和 $k_i$ 均是独立同分布 (i.i.d.) 的随机变量，且满足均值为 0、方差为 1 的标准分布：
$$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \quad \text{Var}(q_i) = \text{Var}(k_i) = 1$$

则在未缩放前，它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 满足：
$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}(q \cdot k) = d_k$$
</div>

### 证明过程：

根据期望的线性性质，由于 $q_i$ 与 $k_i$ 相互独立，其乘积的期望为：
$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0 \times 0 = 0$$

因此，整体内积的期望为：
$$\mathbb{E}[q \cdot k] = \mathbb{E}\left[ \sum_{i=1}^{d_k} q_i k_i \right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0$$

接下来计算单项乘积 $q_i k_i$ 的方差。由于方差的定义为 $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$，我们有：
$$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2$$

由于 $q_i$ 和 $k_i$ 独立，上式可写作：
$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0$$

由于已知 $\text{Var}(q_i) = \mathbb{E}[q_i^2] - (\mathbb{E}[q_i])^2 = 1 \implies \mathbb{E}[q_i^2] = 1$，同理 $\mathbb{E}[k_i^2] = 1$。
带入可得：
$$\text{Var}(q_i k_i) = 1 \times 1 = 1$$

由于各维度分量独立同分布，内积的方差即为各维度分量方差之和：
$$\text{Var}(q \cdot k) = \text{Var}\left( \sum_{i=1}^{d_k} q_i k_i \right] = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k$$

**证明毕。**

<div class="note">
<strong>核心推论：</strong><br/>
由上述证明可知，内积的点积值的方差正比于通道维度 $d_k$。当维度 $d_k$ 很大时（例如在 GPT-3 中，维度 $d_{model} = 12288$），内积的绝对值很容易膨胀得非常大。
</div>

---

## 方差膨胀对 Softmax 的危害

为什么内积结果方差太大，会导致注意力机制崩塌？

我们知道，注意力机制最终是通过 $\text{softmax}$ 激活函数将内积转化为概率分布的。对于一个 $K$ 维的输入向量 $z$，$\text{softmax}$ 的输出为：
$$s_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

当输入 $z$ 的各元素之间差异非常大时（即方差极大），假设其中某一项 $z_m$ 明显大于其他项，那么经过指数放大后，$e^{z_m}$ 会远远主导分母，导致：
$$s_m \approx 1, \quad s_j \approx 0 \quad (\forall j \neq m)$$

此时，$\text{softmax}$ 将退化为一个**独热编码 (One-hot Encoding)** 的硬性最大值选择器。这会导致两个极其严重的灾难：

1. **梯度消失 (Vanishing Gradient)**：由于输出接近 0 或 1，$\text{softmax}$ 函数在此处的导数接近于 0，反向传播时的梯度几乎无法向上传递，导致网络无法收敛。
2. **多模态丧失**：模型被迫只能关注单一的一个 Token，无法将注意力平摊在多个相互关联的词汇上。

通过除以 $\sqrt{d_k}$ 进行**缩放 (Scaling)** 以后，我们成功将内积的方差拉回到了稳定的 $1$：
$$\text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1$$

这就保证了无论模型维度 $d_k$ 膨胀到多大，点积的分布依然稳定，Softmax 的输出梯度饱满，模型得以稳定训练。

---

## 极简代码验证

我们可以通过短短几行 Python (PyTorch) 代码，在计算机上完美重现这一数学结论：

```python
import torch
import torch.nn.functional as F

# 设置大维度
dk = 1024
batch_size = 10000

# 随机生成两个独立同分布的标准正态分布向量
q = torch.randn(batch_size, dk) # 期望 0, 方差 1
k = torch.randn(batch_size, dk) # 期望 0, 方差 1

# 1. 计算未缩放的点积
dot_product = (q * k).sum(dim=-1)
print(f"【未缩放】均值: {dot_product.mean().item():.4f}, 方差 (理论应为 {dk}): {dot_product.var().item():.4f}")

# 2. 计算缩放后的点积
scaled_dot_product = dot_product / (dk ** 0.5)
print(f"【 缩放 】均值: {scaled_dot_product.mean().item():.4f}, 方差 (理论应为 1.0): {scaled_dot_product.var().item():.4f}")
```

这也就是数学推导和工程实践结合的最佳写照！在未来的文章中，我们还将探讨**旋转位置编码 (RoPE)** 与 **FlashAttention** 的代数推导，敬请期待。

---
title: "学习大模型前你需要了解的组件"
summary: "汇总大模型大部分组件、概念"
date: 2026-06-11
tags: ["大模型","基础学习"]
category: "学习指南"
draft: false
---

很多人刚开始学大模型时，最容易陷入一种状态：

+ `Transformer` 听过
+ `Attention` 听过
+ `RMSNorm`、`RoPE`、`Tokenizer` 也都听过

但这些词像散落在桌上的零件，知道名字，却不知道它们为什么会一起工作。

所以这篇文章我不想一上来就推公式，而是先做一张**组件地图**：

> 一次大模型训练，到底有哪些核心组件？它们各自解决什么问题？它们之间又是怎么接起来的？

如果只用一句话概括，一个现代大语言模型大概可以看成这条链：

```text
文本 -> Tokenizer -> Embedding -> Positional Information -> Attention/MLP 堆叠 -> lm_head -> Loss -> 反向传播
```

这条链上的每一个组件，都不是随便放进去的。

---

##### 一、先有一个全局视角：大模型到底在做什么？

一个大语言模型训练时，本质上是在做一件事：

> 给定前面的 token，预测下一个 token。

比如：

```text
我 爱 -> 你
今天 天气 -> 不错
The capital of France is -> Paris
```

所以它不是在“理解世界”之后再生成文字，而是先通过大量训练，学会：

+ 哪些 token 经常一起出现
+ 哪些结构经常共同出现
+ 哪些语义模式在上下文里更合理

而为了完成这件事，它需要下面这些组件。

---

##### 二、Tokenizer：把语言变成数字

这是第一步，也是整个系统的入口。

计算机并不直接理解：

+ 汉字
+ 英文单词
+ 标点
+ 空格

它首先要做的是：

```text
文本 -> token -> token id
```

比如一句话：

```text
"我爱你"
```

可能会被编码成：

```text
[314, 928, 1024]
```

Tokenizer 解决的问题是：

+ 如何切分文本
+ 如何建立词表
+ 每个 token 对应哪个 id

如果你最近在看 tokenizer，可以把它理解成：

> 大模型输入世界的“编码协议”。

它一般在预训练之前就训练好，后续整个模型训练过程基本固定不动。

---

##### 三、Embedding：把离散 id 变成连续向量

Tokenizer 只负责编号，但模型不能直接拿整数做复杂计算。

所以第二步要做的是：

```text
token id -> dense vector
```

假设：

+ 词表大小 `V = 6400`
+ 隐藏维度 `d = 768`

那 embedding 本质上就是一个矩阵：

\[
E \in \mathbb{R}^{V \times d}
\]

每个 token id 对应矩阵里的一行。

如果当前 token id 是 `i`，它的向量就是：

\[
x = E[i]
\]

代码写起来其实非常简单：

```python
import torch
import torch.nn as nn

embed = nn.Embedding(num_embeddings=6400, embedding_dim=768)
input_ids = torch.tensor([[1, 314, 928, 1024]])
hidden_states = embed(input_ids)
print(hidden_states.shape)  # [1, 4, 768]
```

Embedding 解决的问题是：

> 如何把“离散符号”翻译成模型可以在连续空间里计算的向量表示。

---

##### 四、位置编码：Transformer 先天不懂顺序，所以必须补位置信息

Attention 的一个好处是并行，坏处是：

> 它本身不带顺序感。

也就是说，如果你只把 token 向量扔进去，模型天然不知道：

+ 谁在前面
+ 谁在后面
+ 当前 token 距离另一个 token 有多远

所以需要额外注入位置信息。

早期常见的是：

+ sinusoidal positional encoding
+ learned positional embedding

而现代 LLM 非常常见的是：

+ **RoPE（Rotary Position Embedding）**

RoPE 不是简单把一个位置向量加到输入上，而是：

> 直接作用在 Attention 的 `Q` 和 `K` 上，对它们做带位置的旋转。

这意味着位置信息不是“加在 token 上”，而是“写进匹配关系里”。

这一点非常现代，也非常重要。

---

##### 五、Attention：模型真正“看上下文”的地方

Attention 是 Transformer 最核心的组件。

它回答的问题是：

> 当前这个 token，在理解自己时，应该更关注前面哪些 token？

为了做到这一点，每个 token 会先被投影成三种向量：

+ `Q`：Query，我想找什么
+ `K`：Key，我能提供什么
+ `V`：Value，我真正携带的内容

如果输入 hidden states 形状是：

\[
X \in \mathbb{R}^{B \times T \times d}
\]

那么通常会先做线性投影：

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

然后计算 attention score：

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

这条公式值得你反复看，因为它几乎就是 Transformer 的灵魂。

简单理解：

+ `QK^T` 算相似度
+ `softmax` 把相似度变成权重
+ 再用这些权重去加权 `V`

最后得到一个“结合了上下文”的新表示。

---

##### 六、Multi-Head Attention：不是只看一遍，而是从多个角度同时看

如果只有一个 attention head，那模型每次只能用一种方式看上下文。

所以 Transformer 通常会把 hidden size 切成多个 head：

+ 每个 head 各自做一套 Q/K/V
+ 最后再把它们拼起来

这就是 `Multi-Head Attention`。

你可以把它理解成：

> 不是一个人盯着全文看，而是很多个小脑袋各自关注不同模式。

有的 head 可能更关心：

+ 语法依赖
+ 长距离指代
+ 标点边界
+ 重复模式
+ 代码结构

这也是为什么 attention 会这么强。

---

##### 七、Normalization：从 BatchNorm 到 LayerNorm，再到 RMSNorm

训练深层网络时，一个老问题一直存在：

> 层数变深之后，数值分布会越来越不稳定，训练容易发散，梯度也容易出问题。

所以神经网络里一直在想办法做“归一化”。

发展路径大致可以粗略理解成：

1. **BatchNorm**
2. **LayerNorm**
3. **RMSNorm**

---

##### 八、BatchNorm：卷积时代非常成功，但不太适合 LLM

BatchNorm 的核心思想是：

> 对一个 batch 内的激活做归一化。

公式写成：

\[
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\]

再带上可学习参数：

\[
y = \gamma \hat{x} + \beta
\]

这里：

+ `\mu_B` 是 batch 均值
+ `\sigma_B^2` 是 batch 方差

它在 CNN 里极其成功，但在 LLM 里问题不少：

+ 序列模型里 batch 结构更复杂
+ 推理时 batch 行为不如训练稳定
+ 长序列场景不够自然

所以 Transformer 路线后来主要用：

+ LayerNorm

---

##### 九、LayerNorm：按 token 自己归一化

LayerNorm 不再依赖整个 batch，而是：

> 对单个 token 的特征维度做归一化。

公式是：

\[
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
\]

\[
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
\]

\[
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

它比 BatchNorm 更适合 NLP 和 Transformer，因为：

+ 不依赖 batch 统计量
+ 每个 token 自己就能完成归一化
+ 训练和推理更一致

---

##### 十、RMSNorm：现代 LLM 很爱用的轻量版本

很多现代大模型，包括 Llama 系列、Qwen 系列的很多实现，都会使用：

+ **RMSNorm**

它相对于 LayerNorm 的变化非常直接：

> 不再减均值，只看均方根。

公式是：

\[
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
\]

\[
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}
\]

和 LayerNorm 相比，它：

+ 更简单
+ 计算更轻
+ 在大模型里往往足够稳定

这就是它受欢迎的原因。

一个最小实现其实很短：

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

如果你现在刚入门，可以先把它理解成：

> RMSNorm 是 LayerNorm 的一个更省事、更轻的现代版本。

---

##### 十一、Residual Connection：为什么网络可以堆得很深

如果没有残差连接，深层网络非常容易训练困难。

残差连接的思想非常朴素：

\[
y = x + F(x)
\]

也就是说，当前层不是完全重新生成表示，而是在原表示上做修正。

Transformer block 里通常有两次残差：

1. Attention 后一次
2. MLP 后一次

它解决的问题是：

+ 深层训练困难
+ 梯度传不回去
+ 表示更新过于激进

所以它其实是现代深网络能“堆起来”的基础设施之一。

---

##### 十二、MLP / FFN：Attention 负责交流，MLP 负责变换

很多人刚接触 Transformer 时，只盯着 attention，看久了会误以为：

> 模型的能力几乎都来自 attention。

其实不是。

Attention 更像是在做：

+ 信息路由
+ 上下文聚合

而 `MLP` 或 `Feed Forward Network` 更像是在做：

+ 非线性变换
+ 特征重组
+ 表示扩展与压缩

典型结构大概是：

\[
\text{FFN}(x) = W_2 \sigma(W_1 x)
\]

在现代 LLM 里还常常会写成门控版本，比如：

+ SwiGLU
+ GeGLU

这类结构通常比最朴素的两层 MLP 更强。

所以你可以粗略理解：

+ Attention 决定“看谁”
+ MLP 决定“怎么想”

---

##### 十三、MoE：不是每次都让所有参数都干活

如果模型继续做大，一个问题会越来越严重：

> 参数量越来越大，但每个 token 真的需要所有参数都参与计算吗？

于是就出现了：

+ **MoE（Mixture of Experts）**

它的核心思想是：

> 让不同 token 动态路由到不同专家，而不是每次都激活整块 MLP。

这样做的好处是：

+ 总参数可以很大
+ 单次激活参数不必同样大
+ 计算效率和容量可以更平衡

当然，它也会带来新的问题：

+ 路由稳定性
+ 负载均衡
+ 训练更复杂

所以你会在训练里看到一些：

+ `aux_loss`

这常常就是 MoE 路由带来的辅助损失。

---

##### 十四、lm_head：把 hidden states 投影回词表

前面一堆 Attention、Norm、MLP 堆叠之后，模型最后拿到的是：

+ 每个位置的 hidden state

但训练目标是：

> 预测下一个 token 是词表里的哪一个。

所以最后一步要把 hidden state 映射回词表维度。

如果：

+ hidden size 是 `d`
+ vocab size 是 `V`

那么最后一层本质上就是一个线性层：

\[
\text{logits} = hW + b
\]

其中：

\[
W \in \mathbb{R}^{d \times V}
\]

输出的 `logits` 形状就是：

+ `[batch, seq_len, vocab_size]`

这一步的作用非常直接：

> 让模型从“内部语义表示”回到“对整个词表的打分”。

---

##### 十五、Loss：训练时模型到底在学什么？

一个语言模型训练时，不是只判断“对”或“不对”，而是会给词表里每个 token 一个分数，然后通过 `softmax` 变成概率分布。

如果真实下一个 token 是 `y`，而模型预测它的概率是：

\[
p(y \mid x)
\]

那么单个位置的交叉熵损失就是：

\[
\mathcal{L} = -\log p(y \mid x)
\]

如果模型给正确 token 很高概率，loss 就小；  
如果模型几乎不相信正确 token，loss 就大。

这比简单的 `0/1` 更有信息量，因为它告诉模型：

+ 不是只要知道错了
+ 还要知道错得有多离谱

在整段序列上，loss 通常会对所有有效位置求平均。

这就是语言模型训练最核心的目标：

> 让正确下一个 token 的概率越来越高。

---

##### 十六、优化器、学习率、反向传播：真正让参数动起来的部分

有了 loss 之后，还不等于训练完成。

接下来发生的事情是：

1. `loss.backward()`
2. 计算所有参数的梯度
3. 优化器根据梯度更新参数

最常见的优化器之一就是：

+ `AdamW`

它相对朴素 SGD 更适合大模型训练，因为：

+ 自适应学习率
+ 训练更稳定
+ 对不同参数尺度更友好

同时训练里还经常会配：

+ warmup
+ cosine decay
+ gradient clipping
+ mixed precision

这些都不是模型结构本身，但它们决定了训练能不能稳定跑起来。

---

##### 十七、预训练、SFT、DPO、RL：它们不是不同模型，而是不同训练阶段

很多初学者会误以为：

+ 预训练模型
+ 指令微调模型
+ 对齐模型

是三种完全不同的模型。

其实很多时候，它们是：

> 同一个基座模型，在不同阶段接受了不同训练目标。

可以粗略理解成：

**1. 预训练**

+ 大量文本
+ 目标：预测下一个 token
+ 学到语言和知识统计规律

**2. SFT（监督微调）**

+ 多轮对话、指令数据
+ 目标：学会按助手风格回答

**3. DPO / RLHF / RLAIF**

+ 偏好数据或奖励模型
+ 目标：让回答更符合人类偏好

所以“会聊天”并不是预训练天然就有的，而是后续对齐阶段不断塑形的结果。

---

##### 十八、如果把这些组件串起来，一次训练到底发生了什么？

我们现在可以把全局图补完整一点：

```text
原始文本
-> Tokenizer
-> token ids
-> Embedding
-> 加入位置信息 / RoPE作用于Attention内部
-> 多层 Transformer Block
   -> RMSNorm
   -> Multi-Head Attention
   -> Residual
   -> RMSNorm
   -> MLP / MoE
   -> Residual
-> lm_head
-> logits
-> softmax + cross entropy
-> loss
-> backward
-> optimizer.step()
```

这就是一条完整的大模型训练链。

如果你能把这条链真正看顺，后面再去读任何一个大模型项目，很多陌生代码都会突然变得好认。

---

##### 十九、给刚入门时的自己一个建议

如果你现在刚开始学大模型，我觉得最值得优先搞懂的不是“某一个最新 trick”，而是先把下面几件事真正想通：

1. Tokenizer 到底在做什么
2. Embedding 为什么必要
3. Attention 那条公式到底什么意思
4. RoPE 到底加在了哪里
5. RMSNorm 和 LayerNorm 有什么关系
6. lm_head 为什么最后要映射回词表
7. cross entropy loss 为什么不是 `0/1`

因为这些东西一旦通了，后面你再看：

+ Llama
+ Qwen
+ DeepSeek
+ MiniMind

会发现它们虽然细节不同，但骨架其实都在这一套里。

---

##### 二十、阶段性总结

如果一定要把这篇压成一句话，我会这么说：

> 大模型并不是一个神秘黑盒，它本质上是一套把文本编码成向量、通过注意力和前馈网络不断变换表示、最后再投影回词表并通过损失函数学习参数的系统。

而我们平时听到的那些组件名：

+ Tokenizer
+ Embedding
+ RoPE
+ RMSNorm
+ Attention
+ MLP
+ MoE
+ lm_head
+ Loss

其实都只是这个系统里，各自负责不同环节的零件。

把零件的职责弄清楚之后，大模型这件事就不会再只是一堆名词了。

---

如果后面我继续写这个系列，我最想往下展开的三篇会是：

1. `Tokenizer`：现代 tokenizer 到底怎么设计
2. `Attention + RoPE`：Transformer 真正的计算核心
3. `Loss + 训练阶段`：预训练、SFT、DPO、RL 到底有什么本质区别

相关阅读：

+ [TinySeek-Lab: 12_code_first_dense_lm](https://github.com/legend91019/TinySeek-Lab/blob/main/docs/zh/12_code_first_dense_lm.md)

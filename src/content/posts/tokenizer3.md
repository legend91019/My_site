---
title: "Tokenizer(三)，赏析MiniMind的Tokenizer"
summary: "从一个真实小模型项目出发，看看工程里的Tokenizer到底长什么样"
date: 2026-07-10
tags: ["大模型","基础学习"]
category: "学习指南"
draft: false
---

上一篇里，我自己手搓了一个**最简单的 BPE Tokenizer**。那一篇的目标很明确：先把 `merge rules / vocab / encode / decode` 这些最核心的概念讲明白。

但如果我们真的打开一个开源项目，会发现真实世界里的 tokenizer 远远不只是“合并几个字符对”这么简单。

它往往还要处理：

+ 词表大小到底取多少
+ 中英文混合怎么切
+ 特殊 token 怎么设计
+ 对话模板放在哪里
+ tool calling、thinking、vision 这种扩展能力怎么进词表
+ 训练代码到底怎么真正调用 tokenizer

所以这篇我不再自己造一个玩具，而是直接看一个真实项目：

> `MiniMind`

我想回答的问题只有一个：

**一个真实可训练、可推理、可聊天的小模型项目，它的 tokenizer 到底是怎么设计的？**

---

##### 一、先说结论：MiniMind 的 tokenizer 不复杂，但很“工程化”

我先把结论放前面。

MiniMind 这套 tokenizer 的核心配置可以概括成一句话：

+ **BPE**
+ **ByteLevel**
+ **6400 词表**
+ **一大批面向对话、工具调用、多模态预留的特殊 token**
+ **chat template 直接写进 tokenizer 配置里**

也就是说，它不是一个“只负责分词”的小工具，而更像是：

**整个模型输入协议的一部分。**

这一点非常重要。

在上一篇里，我把 Tokenizer 讲成了“文字世界”和“数字世界”之间的翻译器，这个说法没错；  
但到了真实项目里，你会发现 Tokenizer 还顺手承担了另一层职责：

**规定模型到底该怎么说话。**

---

##### 二、我在项目里主要看了哪几个文件

如果你也想自己顺着源码读，最关键的是下面几个文件：

+ `trainer/train_tokenizer.py`
+ `model/tokenizer.json`
+ `model/tokenizer_config.json`
+ `dataset/lm_dataset.py`
+ `README.md` 里 Tokenizer 那一节

它们分别对应：

+ **怎么训练 tokenizer**
+ **训练出来的 tokenizer 底层结构是什么**
+ **特殊 token 和 chat template 怎么配置**
+ **训练数据是怎么真正调用 tokenizer 的**
+ **作者自己为什么这样设计**

也就是说，这不是只读一个 `tokenizer.json` 就能看明白的事情。

Tokenizer 在工程里至少有三层：

1. **切分规则是什么**
2. **配置和协议是什么**
3. **训练代码怎么真正依赖它**

MiniMind 这套设计恰好把三层都暴露得比较清楚，所以特别适合拿来赏析。

---

##### 三、MiniMind 选的不是 SentencePiece，而是 BPE + ByteLevel

先看训练脚本里的核心几行：

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=all_special_tokens
)
```

这几行几乎已经把设计思想写脸上了。

它不是：

+ 纯字符级 tokenizer
+ 传统整词 tokenizer
+ SentencePiece Unigram

而是：

+ **BPE 负责学习合并规则**
+ **ByteLevel 负责把输入先压到 byte 级别来处理**

这有什么意义？

上一篇里我讲最小 BPE 时，是从“字符”开始合并的。那样很好理解，但现实里有一个问题：

**字符集太复杂了。**

英文字符还好，中文、符号、emoji、全角半角、控制字符、各种 Unicode 混进来之后，如果你坚持拿“字符”当最小单位，事情会迅速变脏。

ByteLevel 的思路就很朴素：

**别直接从字符出发，先把文本落到 byte 层。**

这样做的好处是：

+ 最小单位稳定
+ 几乎不会 OOV
+ 中英文混排、奇怪符号、特殊空格都更容易统一处理

所以 MiniMind 这套 tokenizer，虽然名字里还是 BPE，但它其实更接近现代工程里常见的：

**ByteLevel BPE**

这也是为什么在 `model/tokenizer.json` 里，你会直接看到：

```json
"pre_tokenizer": {
  "type": "ByteLevel"
},
"decoder": {
  "type": "ByteLevel"
},
"model": {
  "type": "BPE"
}
```

这三件事合起来，才是它真正的 tokenizer 结构。

---

##### 四、为什么词表只有 6400？

这一点是我觉得 MiniMind 最值得讲的地方之一。

在很多大模型里，词表常常是：

+ 32000
+ 64000
+ 100000+
+ 150000+

但 MiniMind 这里选的是：

> `6400`

第一次看到这个数字，其实会本能地觉得它有点小。

作者在 README 里也直接承认了这件事：  
这套 tokenizer 在中文上的编解码效率，不如更偏中文友好的 tokenizer，比如 Qwen、GLM 这类更大的词表。

那为什么还坚持用 6400？

因为 MiniMind 是一个**小模型项目**。

对于小模型来说，词表大小不只是“分词精不精细”的问题，它还会直接影响参数量。

原因很简单：

+ 输入 embedding 要乘一个 `vocab_size`
+ 输出 lm_head 也要乘一个 `vocab_size`

如果模型隐藏维度是固定的，那么词表越大，这两块参数就越重。

所以对 MiniMind 这种小模型来说，作者更在意的是：

**把 tokenizer 做得足够可用，但不要让词表把 embedding 和输出头撑得太肥。**

这就是一个非常典型的工程权衡：

+ 大词表：压缩率更高，尤其更友好中文
+ 小词表：参数更省，更适合小模型体积控制

MiniMind 选的是后者。

也就是说，它不是在追求“最强 tokenizer”，而是在追求：

**对小模型来说更划算的 tokenizer。**

这和上一篇那个玩具 BPE 最大的不同在于：

上一篇我们关心的是“原理正确”；
这里作者关心的是“整体系统值不值”。

---

##### 五、ByteLevel + 小词表，意味着什么？

README 里提到，这套 tokenizer 在中文文本上大约是：

+ `1.5 ~ 1.7 字符 / token`

在纯英文上大约是：

+ `4 ~ 5 字符 / token`

这个数字本身就很能说明问题。

它告诉我们：

+ 对中文，它切得不算特别激进，一个 token 往往只承载 1 到 2 个字左右
+ 对英文，它又能吃掉更长的片段，所以压缩率明显更高

这正是小词表 ByteLevel BPE 的典型气质：

+ 泛化能力强
+ 几乎不怕奇怪字符
+ 但中文压缩率不会像超大词表那样夸张

所以如果你只看“中文每个 token 能包几个字”，它并不算特别奢侈。
但如果你把目标换成“小模型可训练、可复现、参数受控”，它就突然变得很合理。

---

##### 六、MiniMind 真正有意思的地方，不是 BPE，而是特殊 token 的设计

如果只看到 `BPE + ByteLevel`，这篇文章其实还不够有意思。

真正让我觉得“这已经不是玩具了”的，是它对特殊 token 的设计。

在训练脚本里，作者一开始就塞进去了一长串特殊 token，例如：

```python
"<|endoftext|>", "<|im_start|>", "<|im_end|>"
"<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
"<tool_call>", "</tool_call>"
"<tool_response>", "</tool_response>"
"<think>", "</think>"
```

再往后看，还预留了一堆：

```python
<|buffer1|>, <|buffer2|>, ...
```

这说明什么？

说明作者并没有把 tokenizer 只当成“文本切词器”。

他其实是在提前为这些能力留位置：

+ 对话格式
+ 工具调用
+ 思考链输出
+ 图像/视频/音频相关扩展
+ 未来版本的协议演化

这和很多初学者脑子里的 tokenizer 很不一样。

很多人会觉得 tokenizer 只是：

> 把一句话切成一堆 token

但 MiniMind 告诉你，真实项目里的 tokenizer 更像：

> 先把模型未来可能要说的“语言协议”定义出来，再给它们编号。

这一步一旦做了，后面的训练、推理、工具调用、流式输出就都有了共同语言。

---

##### 七、一个很细的点：`<tool_call>` 和 `<think>` 不是“特殊到不能解码”的那种 special token

这是我觉得 MiniMind 非常细腻的一个地方。

在 `trainer/train_tokenizer.py` 里，`<tool_call>`、`<tool_response>`、`<think>` 这些 token 是加进词表了，但后面作者又手动改了一下 `tokenizer.json` / `tokenizer_config.json` 的标记，让它们不是那种严格意义上“需要在 decode 时跳过”的 special token。

比如在 `tokenizer_config.json` 里能看到：

```json
"25": {
  "content": "<think>",
  "special": false
}
```

这背后的直觉其实很好理解。

像：

+ `<|im_start|>`
+ `<|im_end|>`
+ `<|endoftext|>`

这些 token 更像“底层协议边界”，很多时候你可能希望它们参与控制流程，但不一定希望普通文本输出时原样暴露。

而：

+ `<think>`
+ `<tool_call>`
+ `<tool_response>`

这类 token 更像“模型显式表达结构”的一部分。

如果把它们也处理成那种强 special、decode 时随手吞掉的 token，反而会让调试、日志分析、工具调用协议解析都变麻烦。

所以 MiniMind 这里做了一个很工程化的区分：

+ **有些 token 是真正的边界控制 token**
+ **有些 token 是模型输出结构的一部分，最好能正常保留下来**

这就是“会写 tokenizer”和“只是会调库”的差别。

---

##### 八、在 MiniMind 里，Tokenizer 已经不只是词表，而是 Prompt 协议

如果说上面还只是“词表设计比较认真”，那 `chat_template` 这部分就已经彻底进入系统设计层了。

在 `tokenizer_config.json` 里，MiniMind 直接把整套对话模板写进了 tokenizer 配置。

它大概会把消息组织成这种结构：

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
...
<|im_end|>
<|im_start|>assistant
<think>
...
</think>
...
<|im_end|>
```

工具调用时，还会进一步插入：

```text
<tool_call>
{"name": "...", "arguments": ...}
</tool_call>
```

这说明一个非常重要的事实：

**对 MiniMind 来说，Tokenizer 已经顺手承担了 prompt formatting 的职责。**

也就是说，项目作者不希望每个训练脚本、每个推理脚本、每个 demo 都手写一遍提示词模板。

他希望统一从 tokenizer 里拿：

```python
tokenizer.apply_chat_template(...)
```

这样整个项目就会自然获得：

+ 统一的 system / user / assistant 边界
+ 统一的 tool call 格式
+ 统一的 thinking 格式
+ 统一的生成起点

这件事特别像“把协议内建进 tokenizer”。

所以在真实工程里，Tokenizer 不只是“怎么切词”，还决定了：

**模型训练时到底看见了什么样的对话文本。**

---

##### 九、Tokenizer 甚至直接影响训练标签怎么打

我继续往 `dataset/lm_dataset.py` 看，发现它不是简单做个：

```python
input_ids = tokenizer(text).input_ids
```

而是先：

```python
prompt = tokenizer.apply_chat_template(...)
```

再去编码。

更关键的是，它后面会根据：

```python
tokenizer.bos_token
tokenizer.eos_token
```

去识别 assistant 段落的起止范围，然后只在需要学习的区域上打 label。

换句话说，Tokenizer 的这些设计不是“看起来优雅”而已，而是真的会影响：

+ 哪些 token 参与损失计算
+ 哪些 token 只是上下文
+ assistant 回答从哪里开始学
+ 回答到哪里算结束

所以这里可以得到一个很重要的认识：

**Tokenizer 的配置会直接影响训练监督信号的边界。**

一旦你换掉 tokenizer，哪怕模型结构不变，训练数据的切分和标签位置都可能跟着变。

这也是为什么项目作者在 README 里一直强调：

> 不建议随便重新训练 tokenizer

不是因为训不出来，而是因为：

**Tokenizer 变了，整个训练生态就跟着一起变。**

---

##### 十、MiniMind 给我的最大启发：Tokenizer 其实是模型系统设计的一部分

看完这套实现之后，我最大的感受反而不是“BPE 学到了什么”，而是：

**一个真实项目里的 tokenizer，从来都不是孤立存在的。**

它至少同时连接了四件事：

1. **文本如何切分**
2. **词表如何编号**
3. **对话协议如何组织**
4. **训练标签如何对齐**

上一篇我写最小 BPE 时，Tokenizer 还是一个很“干净”的对象：

+ 训练 merge rules
+ 建 vocab
+ encode / decode

但到了 MiniMind 这里，Tokenizer 已经开始承担系统级职责：

+ 它决定小模型是否能承受当前词表规模
+ 它决定工具调用和 thinking 是否能自然进入训练
+ 它决定 prompt 有没有统一协议
+ 它决定训练脚本怎样精确找到 supervision 的边界

所以如果让我用一句话总结 MiniMind 的 tokenizer：

> 它不是一个“切词器”，而是一套面向小模型训练、对话协议和未来扩展能力的统一入口。

---

##### 十一、回头看上一篇，我会怎么修正自己对 Tokenizer 的理解？

上一篇结束时，我大概会说：

> Tokenizer 就是把文本切成 token，再把 token 变成 id。

这句话依然正确，但现在我会补一句更接近工程现实的话：

> 在真实项目里，Tokenizer 往往还顺手定义了模型该如何组织消息、如何表达结构、如何接入工具，以及训练时哪些 token 应该被学习。

也就是说：

+ 在算法层面，Tokenizer 是分词器
+ 在工程层面，Tokenizer 是协议层

MiniMind 的价值，不在于它用了某个特别新奇的 tokenizer 算法；  
恰恰相反，它用的是一个并不花哨、但很实用的组合：

+ `BPE + ByteLevel`
+ 小词表
+ 明确的特殊 token
+ 内建 chat template

这些东西单独看都不算惊艳，但拼起来之后，就很像一个真正能落地的小模型项目了。

---

##### 十二、如果我要继续往下读 MiniMind，我接下来会看什么？

如果这篇只是“Tokenizer 赏析”的起点，那我下一步最想继续看的是：

+ `chat_template` 和 `tool calling` 的训练数据到底长什么样
+ `<think>` 在 SFT / RL 阶段分别怎么被使用
+ `6400` 词表到底给 embedding 和 lm_head 节省了多少参数
+ 同一段中英混合文本，在 MiniMind tokenizer 下到底是怎么切开的

因为看到这里我已经越来越确认：

**Tokenizer 不只是模型的前处理，而是整个 LLM 项目最底层的一份协议。**

这也是我觉得 MiniMind 很适合拿来学习的原因。

它没有把事情藏得太深，反而把 tokenizer 的训练、配置、模板、训练调用都摊开给你看了。

对于刚从“玩具 BPE”迈向“真实项目”的人来说，这种代码非常有教育意义。

---

参考：

+ `MiniMind/trainer/train_tokenizer.py`
+ `MiniMind/model/tokenizer.json`
+ `MiniMind/model/tokenizer_config.json`
+ `MiniMind/dataset/lm_dataset.py`
+ `MiniMind/README.md`

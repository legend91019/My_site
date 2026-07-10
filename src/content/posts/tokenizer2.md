---
title: "Tokenizer(二)，更现代的Tokenizer"
summary: "BPE几个level的区别，以及现在LLM在使用的tokenizer"
date: 2026-07-10
tags: ["大模型","基础学习"]
category: "学习指南"
draft: false
---

上一篇里，我用一个最小的例子把 `BPE` 的核心直觉讲了一遍：

> 从最小单位开始，反复合并最常见的相邻片段。

但如果我们继续往前走，很快就会发现一个问题：

**BPE 其实不是一种唯一固定的 tokenizer。**

同样叫 BPE，不同项目里“最小单位”可能完全不一样：

+ 有的从字符开始
+ 有的从 byte 开始
+ 有的交给 SentencePiece 直接在原始文本上训练
+ 有的还会额外加上 chat template、tool calling、thinking token

所以这篇我想回答两个问题：

1. **BPE 的几个 level 到底有什么区别？**
2. **现在的 LLM，主流到底在用哪一类 tokenizer？**

---

##### 一、先把一句话说死：现代 LLM 基本都不用“整词分词”了

最早我们很容易把 tokenizer 理解成“切单词”。

比如：

> `hello world`
> ↓
> `["hello", "world"]`

这种整词 tokenizer 的问题在上一篇其实已经说过了：

+ OOV 很严重
+ 词表会爆炸
+ `run / running / runner` 的关联性很弱
+ 中英文混合、多语种、符号、代码都很麻烦

所以现代 LLM 的主流路线，几乎都不是：

+ word-level tokenizer

而是：

+ **subword tokenizer**

也就是说，token 既不是一个字符，也不一定是一个完整单词，而是某种“高频片段”。

这正是 BPE、WordPiece、Unigram 这些方法存在的意义。

---

##### 二、BPE 到底有哪几个“level”？

如果只说 `BPE`，其实信息是不够的。

因为 BPE 只告诉你：

> 我会不断合并高频相邻对。

但它没有告诉你：

> 一开始最小单位是什么？

而这恰恰决定了 tokenizer 的气质。

我自己现在更愿意把 BPE 粗略分成三种常见 level：

1. **Character-level BPE**
2. **Byte-level BPE**
3. **SentencePiece 风格的子词训练（常见 BPE / Unigram）**

这三种思路表面都在做“子词切分”，但实际差异非常大。

---

##### 三、Character-level BPE：最适合教学，但不太适合直接上生产

上一篇我写的最小 BPE，其实就是最典型的：

+ **Character-level BPE**

也就是：

+ 初始 vocab 是字符
+ 再不断合并字符对

比如：

> `l o w e s t`
> ↓
> `lo w e s t`
> ↓
> `low est`

它最大的优点是：

+ 非常直观
+ 容易理解 BPE 的训练过程
+ 拿来教学几乎完美

但它有几个现实问题：

1. **字符集并不稳定**

英文还好，但一到中文、emoji、稀有符号、全角半角、各种 Unicode 混进来，字符集就开始变脏。

2. **跨语言处理麻烦**

你如果把“字符”当最小单位，就天然依赖“字符边界”这件事；但现实文本不只有英语和中文。

3. **工程上不够通用**

工业界更想要一种几乎不怕字符集变化、能覆盖任意文本的最小单位。

所以 Character-level BPE 很适合拿来理解原理，但现代大模型里，直接这么玩的反而不算最多。

---

##### 四、Byte-level BPE：现代 GPT 风格 tokenizer 的核心气质

如果 Character-level BPE 的最小单位是“字符”，那 Byte-level BPE 的最小单位就是：

+ **byte**

也就是：

+ 初始 vocab 先覆盖 256 个 byte
+ 再在 byte 序列上做 BPE 合并

这件事的直觉很重要：

**字符太依赖语言，byte 更底层，也更稳定。**

因为任何文本，无论是：

+ 英文
+ 中文
+ 日文
+ 代码
+ emoji
+ 特殊空格

最后都能落成一串 byte。

所以 Byte-level BPE 的优点非常明显：

1. **几乎不会真正 OOV**

因为最坏情况下，总能退回到 byte 级表示。

2. **语言无关性更强**

它不需要先假设“什么叫一个字符”或者“什么叫一个单词”。

3. **更适合混合文本**

现代 LLM 经常要处理：

+ 中英文混排
+ markdown
+ JSON
+ 代码
+ tool call
+ 各种奇怪分隔符

byte-level tokenizer 面对这些情况更稳。

但它也有代价：

+ 对某些语言，尤其非拉丁文字，压缩率未必最优
+ token 看起来可能更“怪”
+ 空格、换行、前导空白经常会变成切分规则的一部分

所以很多人第一次看 GPT 风格 tokenizer 会觉得奇怪：

> 为什么 `" hello"` 和 `"hello"` 像是两个不同的 token 模式？

答案就在这里：

**空格本身就是建模对象的一部分。**

这不是 bug，而是 byte-level tokenizer 故意保留下来的信息。

---

##### 五、SentencePiece：不先假设“单词边界”，直接在原始文本上训练

如果说 Byte-level BPE 是“从 byte 出发”，那 SentencePiece 的核心思想更像：

> 不依赖外部预分词，直接从 raw text 训练 subword model。

它本身不是单一算法，而更像一个框架，里面常见两条路：

+ SentencePiece BPE
+ SentencePiece Unigram

这和上面的区别在于：

1. **它不要求你先按空格切成单词**

这对中文、日文、韩文这类没有天然空格边界的语言特别友好。

2. **它把空白也显式编码**

你经常会看到类似这样的记号：

> `▁hello`

这里的 `▁` 本质上是在表示“这个 token 前面有一个空格/词边界”。

3. **它对多语种更自然**

因为它不是先依赖英语式的空格分词，再做 BPE。

所以 SentencePiece 这一路非常适合：

+ 多语种模型
+ 非空格分词语言
+ 希望训练流程更统一的项目

它和 Byte-level BPE 没有谁绝对更先进，更多是两种工程取舍：

+ Byte-level BPE 更底层、更稳、更通用
+ SentencePiece 更像“直接在原始文本上学子词”

---

##### 六、所以“BPE 几个 level 的区别”到底区别在哪？

说到底，差别主要在三件事：

**一. 初始最小单位是什么**

+ Character-level：字符
+ Byte-level：byte
+ SentencePiece：更强调从 raw text 学子词，不强依赖外部分词

**二. 是否依赖预分词**

+ 传统英语路线经常先按空格或规则做一些预切分
+ SentencePiece 明确强调可以直接吃原始文本

**三. 空格和边界怎么表示**

+ 有的 tokenizer 把空格当普通字符处理
+ 有的把前导空白作为 token 模式的一部分
+ SentencePiece 会显式把空格/边界编码出来

所以真正决定 tokenizer 气质的，不只是“是不是 BPE”，而是：

+ 它从什么单位开始
+ 它如何处理空格
+ 它是否依赖预分词
+ 它的特殊 token 协议怎么设计

---

##### 七、现在的 LLM 主流到底在用什么？

如果不纠结某一个具体模型，而是看大方向，那我觉得可以很粗暴地总结成一句：

**现代 LLM 主流几乎都在用“子词 tokenizer”，其中最常见的两条路线就是：**

1. **Byte-level BPE / 类 GPT 路线**
2. **SentencePiece 路线（BPE 或 Unigram）**

也就是说，现在已经很少有人认真考虑：

+ 整词 tokenizer 当主流方案

因为大模型面临的问题太复杂了：

+ 多语种
+ 长上下文
+ 工具调用
+ 代码
+ 数学符号
+ markdown / JSON / XML
+ 多模态扩展 token

这些场景都要求 tokenizer 足够稳、足够通用、足够可扩展。

所以今天的 tokenizer 不只是“切词器”，而更像：

**模型输入协议的一部分。**

---

##### 八、现代 tokenizer 和上一篇那个最小 BPE，真正差在哪？

上一篇那个最小 BPE，核心是讲清楚：

+ merge rules
+ vocab
+ encode
+ decode

而现代 tokenizer 在这套骨架之外，又多了几层工程现实：

**1. 最小单位不再只是字符**

它可能是 byte，也可能是 SentencePiece 风格的 raw text 子词学习。

**2. 词表不只是“普通子词”**

还会包含：

+ `<bos>`
+ `<eos>`
+ `<pad>`
+ `<unk>`
+ chat template 用的角色 token
+ `<think>`
+ `<tool_call>`
+ 多模态 token

**3. 它往往内建 prompt 协议**

现代聊天模型常常不是简单 encode 一句文本，而是先：

+ system
+ user
+ assistant
+ tool response

拼成统一格式，再送进 tokenizer。

**4. 它会直接影响训练成本**

同一段文本，tokenizer 切得越碎：

+ 序列越长
+ 显存越贵
+ 训练越慢
+ 推理越贵

所以 tokenizer 从来不是“前处理小细节”，而是会直接影响整个模型系统的效率。

---

##### 九、为什么现在大家几乎都不再纠结“字符级是不是更纯粹”？

因为工程上真正要的不是“概念最纯”，而是“整体最合适”。

字符级当然有它的美：

+ 不容易 OOV
+ 原理直接
+ 词表小

但它的问题同样明显：

+ 序列太长
+ 学语义太慢
+ 对现代长上下文和大规模训练不划算

而现代 subword tokenizer 的价值就在于，它卡在一个很好的中间层：

+ 比字符更有语义
+ 比整词更灵活
+ 比纯词典方法更稳

所以今天的主流选择，不是“最优雅的理论方案”，而是：

**最能兼顾压缩率、泛化能力、工程稳定性和扩展能力的方案。**

---

##### 十、如果让我用一句话总结“更现代的Tokenizer”

我现在会这么总结：

> 更现代的 tokenizer，本质上不是“更会切词”，而是更适合大规模、多语种、长上下文、对话协议和工具调用这些真实工程场景。

所以你可以把 tokenizer 的演化简单看成这条线：

+ 整词 tokenizer：太死
+ 字符级 tokenizer：太碎
+ 子词 tokenizer：刚刚好
+ 现代 tokenizer：在“子词”基础上继续工程化

而所谓“工程化”，其实就是把这些东西一起纳入设计：

+ byte 还是字符
+ raw text 还是预分词
+ 空格怎么编码
+ 特殊 token 怎么设计
+ chat template 怎么接
+ 多模态扩展怎么留口子

这也是为什么今天再看 tokenizer，它已经不是一个边角料组件，而是一个会真正影响：

+ 模型训练效率
+ 上下文长度利用率
+ 多语种表现
+ 工具调用格式
+ 推理接口兼容性

的基础设施。

---

##### 十一、给自己的一个阶段性结论

如果我现在回头看“Tokenizer 是什么”这个问题，我会把答案分成两层：

**算法层面**

+ BPE / WordPiece / Unigram 都是在学子词

**工程层面**

+ 真正决定 tokenizer 体验的，是最小单位、空格处理、词表大小、特殊 token、模板协议这些东西

所以：

+ `BPE` 讲的是“怎么合并”
+ `level` 讲的是“从哪里开始合并”
+ `现代 tokenizer` 讲的是“怎么把这个东西做成真正能服务 LLM 的系统”

这三层不能混在一起看。

---

如果下一篇我继续写，我最想写的反而不是“再解释一个算法”，而是：

> 同一段中文、英文、代码、JSON，用不同 tokenizer 切出来到底会差多大？

因为到那一步，Tokenizer 对训练效率和推理成本的影响就会变得非常直观。


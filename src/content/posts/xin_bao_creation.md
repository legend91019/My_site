---
title: "芯宝是怎么诞生的"
summary: "从桌面启动、意图路由到 RAG、流式生成和长期记忆，拆解芯宝的完整软件机制"
date: 2026-05-21
tags: ["RAG", "检索", "向量数据库", "机器人", "大创项目"]
category: "项目总结"
draft: false
---

> 项目地址：
> [Your-Desktop-dialogue-robot](https://github.com/legend91019/Your-Desktop-dialogue-robot)

![芯宝系统架构图](./images/xinbao_framework.png)

# 一、先给出结论：芯宝是什么

芯宝不是一个把所有能力都塞进大模型的聊天页面，而是一个运行在 Windows 上的本地桌面应用。它把几个职责拆开：

1. 浏览器页面负责交互、语音输入、流式文字显示、表情和音频播放。
2. Python 后端负责 API、会话状态、好感度、路由、检索和提示词组装。
3. 本地模型负责意图分类、文本向量化和候选记忆精排。
4. DeepSeek 云端模型负责最终回答和长期记忆提取。
5. ChromaDB 负责把知识片段和动态记忆持久化，并提供向量查询。

本文只解释软件链路。仓库中的 `hardware_product/` 是面向板端/硬件产品的独立目录；本文不展开唤醒桥接、板端语音环路和硬件通信。

## 1.1 一次回答的核心闭环

~~~text
用户输入
  │
  ▼
FrontEnd/robot.html
  │  POST /api/chat
  ▼
BackEnd/simple.py::handle_chat
  │
  ├─ 好感度更新与 Mood 映射
  ├─ 关键词强制路由 / route_classifier 分类
  ├─ 可选：ChromaDB 召回 + CrossEncoder 精排
  ├─ 拼接系统提示词、短期历史和当前问题
  └─ 请求 DeepSeek 流式生成
       │
       ├─ SSE 首包：好感度和表情变化
       ├─ SSE 文本片段：前端逐段渲染
       ├─ TTS：生成音频并返回 /static/... URL
       └─ 回答结束后启动后台记忆提取线程
~~~

这里有一个重要的时序区别：**回答生成和长期记忆提取不是两个同时运行的回答线程**。后端先完成本次回答、语音和历史记录，再通过 `threading.Thread` 异步启动 `extract_and_save_memory()`；记忆提取失败不会回滚当前回答。

# 二、项目结构：每个目录负责什么

~~~text
IntelliChat-Platform/
├─ FrontEnd/                 # HTML/CSS/JavaScript 交互界面
│  └─ robot.html
├─ BackEnd/                  # Flask 路由和运行时服务
│  ├─ simple.py              # 后端主编排层
│  ├─ memory_admin.py        # 长期记忆增删改查
│  ├─ tts_engine.py          # Edge-TTS / Index-TTS 适配
│  └─ audio_player.py        # 本机音频播放
├─ utils/
│  ├─ Classifier/            # 轻量意图路由器
│  └─ Retriever/             # knowledge.md 切块与向量检索器
├─ models/
│  ├─ embedding/             # SentenceTransformer 向量模型
│  └─ reranker/              # CrossEncoder 精排模型
├─ assets/classifier/
│  └─ route_classifier.joblib# 已训练的路由分类器
├─ knowledge.md              # 静态知识库源文件
├─ dynamic_keywords.txt      # 后台记忆提取出的动态触发词
├─ config.example.json       # 配置模板
├─ runtime_paths.py          # 安装资源与用户数据路径
├─ desktop_launcher.py       # 启动后端并打开桌面窗口
├─ startup_checks.py         # GPU、模型和运行环境检查
├─ tools/                    # 模型下载、构建和 Index-TTS 服务脚本
├─ tests/                    # 单元测试和发布检查
└─ hardware_product/         # 独立的板端/硬件产品代码
~~~

几个关键边界如下：

| 模块 | 主要职责 | 不负责什么 |
| --- | --- | --- |
| `FrontEnd/robot.html` | 输入、SSE 解析、聊天渲染、表情、音频播放 | 不执行向量检索和模型推理 |
| `BackEnd/simple.py` | 编排一次请求的完整生命周期 | 不实现 Embedding 或 CrossEncoder 内部算法 |
| `utils/Classifier` | 输出 `pred=0/1` 的路由判断 | 不生成最终回答 |
| `utils/Retriever` | 读取 `knowledge.md`、切块、入库并返回静态检索上下文 | 不负责 DeepSeek 生成 |
| `BackEnd/memory_admin.py` | 长期记忆的列表、添加、修改、删除 | 不决定一条记忆是否值得自动保存 |
| `BackEnd/tts_engine.py` | 文本清洗、TTS 引擎选择、生成音频 | 不决定回答内容 |

# 三、程序如何启动

## 3.1 桌面启动器

普通用户启动的是 `desktop_launcher.py`。它的主要顺序是：

1. 调用 `startup_checks.run_all(project_root())`，检查模型、GPU 和运行环境。
2. 通过 `find_free_port()` 为本机地址 `127.0.0.1` 找一个可用端口。
3. 启动子进程：`python -m BackEnd.simple --host 127.0.0.1 --port <port>`。
4. 轮询 `/`，直到后端可以访问。
5. 调用 `webview.create_window("芯宝 Xinbao", url, ...)` 打开桌面窗口。
6. 窗口关闭后终止后端子进程。

因此，桌面窗口本质上是一个本地 Web UI；前端使用相对路径访问同一个本地 Flask/Waitress 服务，不需要把 API 地址写死。

## 3.2 后端初始化

`BackEnd/simple.py` 在启动时执行 `init_model()`，一次性准备共享对象：

~~~text
init_model()
  ├─ load_route_classifier(assets/classifier/route_classifier.joblib)
  ├─ chromadb.PersistentClient(path=%APPDATA%/Xinbao/chroma_db)
  ├─ get_or_create_collection("qbit_memory")
  ├─ SentenceTransformer(models/embedding)
  ├─ create_rag_retriever(knowledge.md, embed_model, collection)
  └─ CrossEncoder(models/reranker)
~~~

模型和数据库连接放在全局变量 `embed_model`、`collection`、`reranker_model` 中，后续请求重复使用，避免每次聊天重新加载几百 MB 的模型。

服务就绪后由 Waitress 提供 WSGI 服务。健康检查接口 `/api/health` 在 `MODEL_READY=True` 前返回 `503`，模型和检索器初始化完成后才返回正常状态。

## 3.3 数据在哪里保存

安装目录中的代码和模型是只读资源；用户可写数据通过 `runtime_paths.py` 放在 `%APPDATA%\Xinbao\`：

- `config.json`：主人称呼、身份、当前状态、API Key 和语音设置。
- `favorability.json`：当前好感度分数。
- `chroma_db/`：ChromaDB 持久化数据。
- `uploads/`：上传文件。
- `static/audio/`：运行时生成的语音文件。
- `logs/`：后端运行日志。

`dynamic_keywords.txt` 当前由 `BackEnd/simple.py` 按项目相对路径写入仓库根目录，因此它与 `%APPDATA%\Xinbao` 下的用户数据不是同一类路径。

# 四、一次聊天请求的完整过程

## 4.1 前端捕获输入

用户可以打字，也可以通过浏览器 `Web Speech API` 把语音转成文字。`robot.html::sendMessage()` 做三件事：

- 检查 `isTyping`，防止上一条消息还在流式返回时重复提交。
- 先把用户消息渲染到页面，再创建一个带打字光标的机器人气泡。
- 使用 `fetch('/api/chat', { method: 'POST', body: { message } })` 发起请求。

发送前的短音频播放是浏览器自动播放策略的兼容处理：先用静音音频尝试解锁后续音频通道，不是后端语音生成的一部分。

## 4.2 后端入口与快速分支

`handle_chat()` 首先解析 JSON 或表单中的 `message`。如果消息为空，返回 `400`。

接着读取 `favorability.json`。如果用户直接询问“好感度”“喜欢我吗”等问题，后端不调用 DeepSeek，而是根据当前分数直接返回固定文本，并把这次问答写入进程内 `chat_history`。

这条快速分支体现了一个工程取舍：能由本地状态确定的回答，不必消耗一次云端 API 请求。

## 4.3 好感度更新

普通消息会扫描两组词表：

~~~python
add_words = ["乖", "真棒", "厉害", "太聪明了", "好可爱", "超可爱", "爱你", "喜欢你", "贴贴", "抱抱", "摸摸头", "揉揉头"]
sub_words = ["笨", "讨厌", "很烦", "坏", "傻", "闭嘴", "滚", "走开", "不理你", "没用", "差劲"]
~~~

命中加分词时增加 3 分，命中减分词时减少 5 分；代码使用 `if ... elif`，同一条消息最多执行一种变化。`save_favorability()` 用 `clip` 将结果限制在 `[0,100]`：

$$
F_{t+1}=\operatorname{clip}(F_t+\Delta F_t,0,100)
$$

其中：

$$
\Delta F_t=\begin{cases}
+3, & \text{命中加分词}\
-5, & \text{否则命中减分词}\
0, & \text{都未命中}
\end{cases}
$$

分数再映射为 Prompt 中的 Mood：

| 分数 | Mood 提示 |
| --- | --- |
| `F >= 80` | 软萌、撒娇、非常粘人 |
| `51 <= F < 80` | 阳光可爱、温柔回应 |
| `31 <= F <= 50` | 小傲娇、回答简洁 |
| `0 <= F <= 30` | 生气、傲娇、委屈但不恶毒 |

Mood 不是独立的情绪模型，而是一个离散区间门控函数：

$$
M(F)=\begin{cases}
M_1, & F\ge 80\
M_2, & 50<F<80\
M_3, & 30<F\le 50\
M_4, & F\le 30
\end{cases}
$$

## 4.4 双引擎路由

系统先调用 `classifier.predict([user_message])` 得到分类结果，再读取关键词池进行强制修正。

### 第一层：规则强制 RAG

关键词池由两部分组成：

1. `config.json` 中的 `routing_settings.force_rag_keywords`，没有配置时使用默认词表，例如“芯宝、开发、记得、喜欢、谁、以前”等。
2. `dynamic_keywords.txt` 中由后台记忆提取器追加的实体词。

当消息长度大于 1 且包含任一关键词时，代码把 `pred` 强制设为 `1`。这相当于为分类器增加了一道高召回规则：涉及芯宝身份、用户历史、偏好或动态记忆的表达，优先进入 RAG。

### 第二层：轻量分类器

没有命中规则时，使用 `assets/classifier/route_classifier.joblib` 加载的 `RouteClassifier`。它保持统一的 `predict()` 接口，返回：

- `pred = 0`：自由闲聊，不主动检索。
- `pred = 1`：需要检索，进入 RAG 增强模式。

`RouteClassifier` 还包含少量后处理：天气闲聊短语强制归为 `0`，查看、提醒、设备、历史记忆等操作词强制归为 `1`。

可以把路由写成：

$$
R(q)=\begin{cases}
1, & |q|>1\ \land\ q\text{ 命中规则关键词}\
C(q), & \text{否则}
\end{cases}
$$

其中 `C(q)` 是 `RouteClassifier` 的预测结果。

# 五、RAG：静态知识和动态记忆如何进入回答

芯宝的 RAG 有两个来源，但共用一个 ChromaDB collection `qbit_memory`：

- **静态知识**：`knowledge.md`，描述项目设定、团队、世界观和固定资料。
- **动态记忆**：用户对偏好、习惯和经历的长期陈述，例如“主人最喜欢吃三文鱼”。

两者都保存为文档、向量和 metadata；metadata 中可区分 `type`、`source`、`title`、`timestamp` 和 `chunk_index`。

## 5.1 knowledge.md 的切块与增量入库

`utils/Retriever/retriever.py::create_rag_retriever()` 做如下处理：

1. 按标题和空行分割 Markdown 块，记录当前章节标题。
2. 再按 `。！？` 切分句子。
3. 当当前块超过约 300 个字符时，生成一个知识 chunk。
4. 用 `md 文件名 + chunk_index` 的 MD5 前 12 位作为稳定 ID。
5. 查询现有 collection，只对新 ID 进行 Embedding 和 `upsert`。

因此，重复启动不会重复向量化已有知识；知识源发生新增时，只计算新增块。

静态检索器返回的 `retrieve(question)` 默认 `top_k=2`，它把召回的片段格式化为“类型、来源、章节、正文”。在 `init_model()` 中，静态检索器和聊天动态检索使用的是同一个 collection，所以静态检索结果也可能来自已经写入的动态记忆。

## 5.2 动态记忆的 Top-10 → Top-3 漏斗

当路由结果为 `pred=1` 时，`handle_chat()` 除了调用静态检索器，还直接对同一个 collection 执行动态候选筛选：

### 第一步：Embedding

用户问题经过 `SentenceTransformer.encode(..., normalize_embeddings=True)` 得到归一化向量：

$$
\mathbf{v}=\frac{\mathbf{u}}{\|\mathbf{u}\|_2},\qquad \|\mathbf{v}\|_2=1
$$

若原始句子经过 Transformer 后得到 token 隐藏状态 $h_1,\ldots,h_n$，常见的句向量可以抽象为池化结果：

$$
\mathbf{u}=\operatorname{Pool}(h_1,\ldots,h_n)
$$

具体池化由本地 SentenceTransformer 模型封装；当前业务代码只显式要求归一化输出。

### 第二步：ChromaDB 召回 10 个候选

代码执行：

~~~python
query_emb = embed_model.encode([user_message], normalize_embeddings=True).tolist()[0]
results = collection.query(query_embeddings=[query_emb], n_results=10)
~~~

返回结果按 ChromaDB 的距离排序。当前代码保留 `dist < 1.5` 的候选：

$$
\mathcal{C}(q)=\{d_i\mid D(q,d_i)<1.5\}
$$

这里的 `1.5` 是业务层初筛阈值。本文不把 ChromaDB 的底层索引实现（例如具体图索引配置）冒充为本项目自定义算法；项目只通过 ChromaDB API 使用查询结果。

如果向量已归一化且距离采用平方 L2，则有：

$$
\|\mathbf{q}-\mathbf{m}\|_2^2
=\|\mathbf{q}\|_2^2+\|\mathbf{m}\|_2^2-2\mathbf{q}\cdot\mathbf{m}
=2(1-\cos\theta)
$$

在这一理想条件下，`D < 1.5` 对应 `cos(theta) > 0.25`。这说明初筛故意留出较宽候选集，避免仅靠字面相似度漏掉语义相关记忆；但最终距离定义仍由 ChromaDB collection 的配置决定。

### 第三步：CrossEncoder 精排

候选文档与用户问题组成成对输入：

~~~python
pairs = [[user_message, doc] for doc, meta in candidate_docs]
scores = reranker_model.predict(pairs)
~~~

CrossEncoder 的概念输入可以写为：

$$
\text{[CLS]}\ q_1\ldots q_m\ \text{[SEP]}\ d_1\ldots d_k\ \text{[SEP]}
$$

问题和文档在同一次 Transformer 编码中交互，而不是先分别编码后只计算两个固定向量的相似度。自注意力的基本形式为：

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

模型输出一个相关性分数，代码随后按分数降序排列，只保留 `score > 0` 且最多 3 条：

$$
\mathcal{R}(q)=\operatorname{Top3}\{d_i\in\mathcal{C}(q)\mid s(q,d_i)>0\}
$$

每条保留下来的动态记忆会附带 metadata 中的时间戳，形成：

~~~text
[2026-05-20 18:30:00] 主人最喜欢吃三文鱼
~~~

## 5.3 Prompt 融合

RAG 分支最终将以下信息拼成一个发送给 DeepSeek 的 `final_prompt`：

1. 芯宝名称和角色设定。
2. 好感度数值和 Mood。
3. 当前真实时间，用于解释“今天、昨天、上周”。
4. 静态知识检索结果。
5. 带时间戳的动态记忆。
6. 最近 10 条进程内对话历史。
7. 当前用户问题。

提示词还要求模型：使用动态记忆时添加来源脚注；私人信息查不到时明确说不知道；通用历史、文学和科学知识不必受本地资料限制。

闲聊分支不执行上下文检索，只拼接角色、好感度、时间、最近 10 条历史和当前问题。

# 六、流式回答、表情和语音

## 6.1 SSE 首包和文本流

后端使用 `Response(stream_with_context(generate_stream()), mimetype='text/event-stream')` 返回 SSE。`generate_stream()` 的顺序是：

1. 先发送 `{favorability, change}`，前端收到 `up` 或 `down` 后立即切换机器人表情。
2. 以 `stream=True` 请求 DeepSeek，逐行读取 `data: ...`。
3. 从 `choices[0].delta.content` 取出增量文本，逐段发送为 `{chunk: content}`。
4. DeepSeek 返回 `[DONE]` 后结束文本循环。
5. 如果本轮有好感度变化，把提示文本作为最后一个 chunk 追加。

前端不能假设一次网络读取就是一条完整 SSE 消息，因此使用：

~~~javascript
buffer += decoder.decode(value, { stream: true });
const lines = buffer.split('\n');
buffer = lines.pop();
~~~

最后一行保留在 `buffer` 中，等下一次读取拼接，避免 JSON 被网络分片截断。

## 6.2 TTS 和播放兜底

回答完成后，后端对文本执行 `sanitize_tts_text()`：去掉 `[表情]`、括号动作、Markdown 标记和不适合播报的符号；再用 `limit_tts_text()` 限制句数和最大字符数。

随后按配置选择：

- `edge_tts`：生成 `.mp3`。
- `indextts`：请求本地 Index-TTS 服务生成 `.wav`；失败时按配置回退到 Edge-TTS。

生成前会清理超过 180 秒的旧音频。后端先尝试 `play_audio_file()` 在本机播放，再把 `/static/<filename>` 放进 `done` 事件；前端收到后再次设置 `audio` 的 `src` 并播放，作为浏览器侧兜底。

最终结束事件大致包含：

~~~json
{
  "done": true,
  "timestamp": "2026-05-21 20:00:00",
  "audio_url": "/static/reply_ab12cd34.mp3",
  "local_audio": {"played": true}
}
~~~

# 七、长期记忆：回答之后的异步闭环

回答和音频完成后，后端追加本次用户消息与机器人回答到 `chat_history`，然后启动：

~~~python
threading.Thread(
    target=extract_and_save_memory,
    args=(user_message,)
).start()
~~~

`extract_and_save_memory()` 会再次调用 DeepSeek，但使用的是专门的提取提示词：

~~~text
陈述句 | 实体1,实体2
~~~

例如：

~~~text
主人最喜欢吃三文鱼 | 日料,三文鱼
~~~

如果模型返回“无”，或者返回内容过长，函数直接放弃保存。有效结果会经历以下步骤：

1. 将整段提取结果做 Embedding，并用其 MD5 前 12 位作为记忆 ID。
2. 将文本、向量和 `type=user_preference`、`source=dynamic_memory`、`timestamp` 等 metadata `upsert` 到 `qbit_memory`。
3. 把长度大于 1 的实体词追加到 `dynamic_keywords.txt`。

于是下一次对话会形成闭环：

~~~text
用户说出长期信息
  → DeepSeek 提取陈述句和实体词
  → ChromaDB 保存向量记忆
  → dynamic_keywords.txt 增加强制触发词
  → 后续提问更容易进入 RAG
  → 向量召回 + CrossEncoder 精排
  → Prompt 使用带时间戳的记忆
~~~

这个机制是“自动记忆候选写入”，不是无条件永久相信模型输出。项目同时提供 `/api/memories` 管理接口，前端可以查看、手动添加、修改和删除动态记忆。

# 八、主要 API 和状态边界

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/api/health` | GET | 返回后端、模型和 GPU 就绪状态 |
| `/api/settings` | POST | 保存主人设定和 DeepSeek API Key |
| `/api/settings/status` | GET | 只返回是否存在 API Key |
| `/api/chat` | POST | 执行完整聊天链路并返回 SSE |
| `/api/upload` | POST | 保存上传文件并写入一条固定处理结果 |
| `/api/history` | GET/DELETE | 读取或清空进程内聊天历史 |
| `/api/memories` | GET/POST | 查看或手动添加长期记忆 |
| `/api/memories/<id>` | PUT/DELETE | 修改或删除单条长期记忆 |

需要区分三种状态：

- **短期记忆**：`chat_history` 是 Python 进程内列表，只保留最近消息供 Prompt 使用；后端重启后清空。
- **长期记忆**：ChromaDB 中的文档、向量和 metadata，重启后仍可查询。
- **配置状态**：`config.json` 和 `favorability.json` 单独持久化，不等同于对话历史。

另外，当前 `/api/upload` 只完成文件保存和固定回复，并没有把任意上传文件自动解析、切块并加入 RAG；这是现阶段的功能边界。

# 九、答辩时可以怎样概括

芯宝的核心设计不是“让大模型记住一切”，而是把不同类型的状态放在不同层：

1. 用规则和轻量分类器决定是否值得检索，减少无意义的向量查询和云端上下文。
2. 用本地 Embedding 做高召回，再用 CrossEncoder 做低数量、高精度的重排。
3. 把静态知识、动态记忆、短期历史、好感度和当前时间分层拼入 Prompt。
4. 用 SSE 让文字、表情和语音在用户可感知的时间内逐步到达。
5. 在回答完成后异步提取长期记忆，避免记忆写入拖慢当前响应。

因此，芯宝是一个“本地状态管理 + 本地检索模型 + 云端生成模型 + 桌面交互”的组合系统。它的可解释性来自明确的路由规则、可追踪的 metadata、可管理的记忆接口和可以从源码复现的请求时序。

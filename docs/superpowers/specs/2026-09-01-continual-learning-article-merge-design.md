# 持续学习文章合并设计

## 目标

将两篇 AIGC 生成的持续学习文章合并、精简为一篇，同时保持作者亲自撰写的 `orthogonal_lora.md` 完全不变。

## 文件处理

- 保留并重写 `src/content/posts/llm_continual_learning_foundations.md`，作为合并后的唯一文章入口。
- 删除 `src/content/posts/orthogonal_lora_continual_learning.md`，避免主题重复。
- 不修改 `src/content/posts/orthogonal_lora.md` 的正文、元数据或文件名。

## 内容结构

合并后的文章按一条由基础到问题的主线组织：

1. 持续学习、灾难性遗忘与稳定性—可塑性矛盾。
2. replay、正则化、参数隔离和子空间约束的基本路线。
3. LAMOL、LFPT5、Progressive Prompts 与 O-LoRA 的方法演进。
4. 正交与子空间方法的后续发展，以及双线性 LoRA 可能绕过约束的问题。
5. 方法选择、局限与开放问题。

实验内容只保留足以支撑判断的定性结论或少量代表性数字；删除训练超参数、逐项数据集介绍、大型结果表和详细消融复述。

## 元数据

- 合并文章日期设为 `2026-07-29`，即 `orthogonal_lora.md` 日期 `2026-07-30` 的前一天。
- 标题、摘要、标签和分类调整为能覆盖“持续学习基础 + 正交 LoRA 问题”的统一表述。
- 文章保持非草稿状态。

## 验收标准

- 合并后的文章可独立阅读，段落之间有明确过渡，无“上一篇/下一篇”残留指向。
- 不再详细展开实验设置，且不引入无法由原两篇文章支撑的新结论。
- `orthogonal_lora.md` 在最终 diff 中没有变化。
- Astro 生产构建成功，Markdown、公式和 frontmatter 无报错。

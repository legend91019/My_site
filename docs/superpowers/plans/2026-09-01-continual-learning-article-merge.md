# 持续学习文章合并实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将两篇 AIGC 持续学习文章压缩合并为一篇，并保持 `orthogonal_lora.md` 不变。

**Architecture:** 以 `llm_continual_learning_foundations.md` 作为唯一发布入口，重写其 frontmatter 和正文，将基础方法与正交 LoRA 研究线组织成一篇短而连贯的综述；删除重复的 `orthogonal_lora_continual_learning.md`。最后用 Git diff 和 Astro 构建验证内容与站点完整性。

**Tech Stack:** Astro content collections, Markdown, KaTeX-compatible LaTeX, npm。

## Global Constraints

- `orthogonal_lora.md` 不得修改。
- 合并文章日期必须为 `2026-07-29`。
- 实验只保留少量代表性数字或定性结论，不保留详细训练超参、长表格和逐项消融。
- 不引入原两篇文章无法支撑的新结论。
- 删除重复文章文件，避免同主题重复发布。

---

### Task 1: 重写合并文章

**Files:**
- Modify: `src/content/posts/llm_continual_learning_foundations.md`
- Reference only: `src/content/posts/orthogonal_lora.md`
- Reference only: `src/content/posts/orthogonal_lora_continual_learning.md`

**Interfaces:**
- Consumes: 两篇 AIGC 文章现有的公式、论文链接、方法比较和结论。
- Produces: 一篇日期为 `2026-07-29`、可独立阅读、实验精简的持续学习综述。

- [ ] **Step 1: 读取并核对可复用内容**

确认原文章中的方法名称、论文链接、关键公式和少量代表性结果；跳过 `orthogonal_lora.md` 的正文改动。

- [ ] **Step 2: 写入统一 frontmatter**

使用以下字段：

```yaml
title: "语言模型持续学习：从灾难性遗忘到正交 LoRA"
summary: "从 replay、prompt 和 LoRA 出发，梳理持续学习中的子空间隔离、参数保护与双线性优化失稳。"
date: 2026-07-29
tags: ["大模型", "持续学习", "LoRA", "灾难性遗忘"]
category: "论文解读"
draft: false
```

- [ ] **Step 3: 按五段主线重写正文**

正文依次覆盖：问题与指标；方法地图；LAMOL/LFPT5/Progressive Prompts/O-LoRA；Sculpting Subspaces/OA-Adapter/OLieRA/ELLA/SFOR；横向比较与开放问题。每个方法保留“解决什么、核心机制、限制、实验一句话结论”四类信息。

- [ ] **Step 4: 清理重复指向与实验冗余**

删除“上一篇/下一篇”式串联、详细训练超参、逐任务结果表和大段消融；确保标题层级、公式和参考链接仍完整。

### Task 2: 删除重复文章并验证不可变文件

**Files:**
- Delete: `src/content/posts/orthogonal_lora_continual_learning.md`
- Verify unchanged: `src/content/posts/orthogonal_lora.md`

**Interfaces:**
- Consumes: Task 1 完成的合并文章。
- Produces: 只保留一篇 AIGC 合并文章，且亲写文章内容哈希不变。

- [ ] **Step 1: 记录亲写文章校验值**

运行 `Get-FileHash src/content/posts/orthogonal_lora.md -Algorithm SHA256`，编辑前后比较输出。

- [ ] **Step 2: 删除重复文件**

使用补丁删除 `src/content/posts/orthogonal_lora_continual_learning.md`，不触碰 `orthogonal_lora.md`。

- [ ] **Step 3: 检查文章残留引用**

运行 `rg -n "orthogonal_lora_continual_learning|上一篇|下一篇" src/content/posts`，确认没有重复 slug 或不合时宜的串联文字。

### Task 3: 构建与差异验证

**Files:**
- Verify: `src/content/posts/llm_continual_learning_foundations.md`
- Verify: `src/content/posts/orthogonal_lora.md`

**Interfaces:**
- Consumes: 已合并并删除重复文件的内容目录。
- Produces: 构建成功、frontmatter 合法、亲写文章无变化的最终工作树。

- [ ] **Step 1: 运行内容检查**

运行 `rg -n "TBD|TODO|待补" src/content/posts/llm_continual_learning_foundations.md`，预期无输出；运行 `git diff --check`，预期退出码 0。

- [ ] **Step 2: 运行 Astro 生产构建**

运行 `npm run build`，预期 Astro 退出码 0 且无 content-schema 或 Markdown 错误。

- [ ] **Step 3: 检查最终 diff**

运行 `git status --short` 与 `git diff --stat`，确认只包含计划中的合并文章、删除文件及必要文档；再次比较 `orthogonal_lora.md` 的 SHA256 与 Task 2 Step 1 记录一致。

- [ ] **Step 4: 提交实现**

```bash
git add src/content/posts/llm_continual_learning_foundations.md src/content/posts/orthogonal_lora_continual_learning.md
git commit -m "content: merge continual learning articles"
```

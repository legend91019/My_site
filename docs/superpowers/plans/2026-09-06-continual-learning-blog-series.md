# Continual Learning Blog Series Implementation Plan

**Goal:** 将持续学习博客整理为“总体综述、PEFT/LoRA 专题、BOD 论文解读”三篇互相链接的中文文章。

**Architecture:** 保持 Astro content collection 现有 Markdown/frontmatter 约定。第一篇提供领域地图和协议；第二篇沿正交 LoRA 的空间约束主线补充近期方法；第三篇解释当前大修稿的 BOD、SFOR、WRP、O-LoRA+BSR，并明确实验结论与待补证据的边界。

**Tech Stack:** Astro, Markdown, KaTeX/现有内容渲染器。

## Tasks

- [x] 重写 `src/content/posts/llm_continual_learning_foundations.md`，覆盖定义、任务、指标、协议、数据集、方法谱系和阅读清单。
- [x] 扩展 `src/content/posts/orthogonal_lora.md`，统一 `ΔW=sBA` 符号，补充近期 PEFT/全参方法并链接第三篇。
- [x] 创建 `src/content/posts/on_the_instability_of_orthogonal_lora.md`，按论文解读结构介绍 BOD、SFOR、WRP、O-LoRA+BSR、实验协议和证据边界。
- [x] 运行 `npm run build`，修正内容集合或 Markdown 编译问题。

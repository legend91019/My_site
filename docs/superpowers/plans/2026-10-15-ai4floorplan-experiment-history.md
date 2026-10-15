# AI for Floorplan 实验探索历程扩写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `ai4floorplan2.md` 重写为一篇按时间顺序、证据边界清楚且适合 floorplan 初学者阅读的完整实验探索记录。

**Architecture:** 以 2026-09-24 至 2026-10-15 的实验时间线为主轴，把每项研究统一组织成“问题—设置—结果—边界—转向”。第一篇文章承担基础概念，本文只解释实验专用术语，并以 canonical research reports 为事实来源、原始 report 为必要复核来源。

**Tech Stack:** Astro content collection、Markdown/MDX-compatible Markdown、项目现有 npm 构建与测试脚本。

## Global Constraints

- 不重复 `ai4floorplan.md` 已系统介绍的 floorplan、HPWL、B\*-tree、官方评分和约束基础。
- 不把计划、推测或未完成实验写成已验证结论。
- 不把 offline oracle / exact best-of-K 写成可部署收益。
- 保留当前 frontmatter 与 `image-1.png` 引用的站点兼容性。
- 新术语首次出现时必须用初学者可理解的中文就地解释。

---

### Task 1: 建立时间线与证据口径

**Files:**
- Read: `D:/Desktop/Research/ai_for_eda/FloorSet/docs/research/*.md`
- Read: `D:/Desktop/Research/ai_for_eda/FloorSet/iccad2026contest/workspace/experiments/adaptive_heuristic_scheduler/reports/conclusions.md`
- Read: `D:/Desktop/Research/ai_for_eda/FloorSet/.worktrees/learned-order-pilot/docs/research/*.md`
- Modify: `D:/Desktop/ai_create_blog/src/content/posts/ai4floorplan2.md`

**Interfaces:**
- Consumes: canonical research conclusions and current blog text.
- Produces: a chronological section map with one evidence status per experiment.

- [ ] **Step 1: Fix the experiment order**

Use this order: region rebuild → one-step mutation → trajectory → Macro Agent v0 → constructive ordering → proxy lookahead → global-gap diagnosis → coordinate refinement → adaptive scheduler → reference-structure oracle → learned order → selector incident.

- [ ] **Step 2: Fix status labels**

Use only these meanings: `负结果` for completed controlled negative evidence, `阶段性信号` for incomplete or non-generalized evidence, `PASS` for a pre-registered stage gate, and `INCOMPLETE` when upstream guards stopped the experiment.

- [ ] **Step 3: Record numerical claims**

Copy only numbers supported by canonical reports; retain their denominator and comparison arm, such as `18/48 states` rather than an unlabeled `37.5%`.

### Task 2: Rewrite the article

**Files:**
- Modify: `D:/Desktop/ai_create_blog/src/content/posts/ai4floorplan2.md`

**Interfaces:**
- Consumes: the section order and claim rules from Task 1.
- Produces: the complete publishable Markdown article.

- [ ] **Step 1: Rewrite the opening**

Link the first article, explain that this article is an experiment diary, and introduce the recurring diagnostic question: whether good candidates exist and whether a deployable selector can identify them.

- [ ] **Step 2: Expand the six existing experiment sections**

Add motivation, control arms, sample sizes, key numbers, evidence boundaries and transition logic without duplicating the first article's foundational tutorial.

- [ ] **Step 3: Add the October 12–15 experiments**

Cover global-gap diagnosis, coordinate regression, scheduler study, reference oracle, learned proposer and the interrupted selector run, with explicit distinction between offline upper bound and realized deployable gain.

- [ ] **Step 4: Add synthesis**

End with a compact experiment ledger and the current research position: local selection is mostly saturated; executable global-order proposal is promising; label-free selection and fresh-case generalization remain open.

### Task 3: Verify content and site integration

**Files:**
- Verify: `D:/Desktop/ai_create_blog/src/content/posts/ai4floorplan2.md`
- Verify: `D:/Desktop/ai_create_blog/src/content/posts/ai4floorplan.md`

**Interfaces:**
- Consumes: the rewritten article.
- Produces: a buildable, internally consistent article with no unsupported claim or broken asset reference.

- [ ] **Step 1: Run textual audits**

Run:

```powershell
rg -n "TBD|TODO|待补|8\.06|INCOMPLETE|oracle|selector|trajectory" src/content/posts/ai4floorplan2.md
```

Expected: no placeholders; every `8.06%` occurrence says offline/exact/best-of-K; selector incident is labeled `INCOMPLETE`; trajectory is not described as a completed 100-case positive gate.

- [ ] **Step 2: Run the project checks**

Run:

```powershell
npm test
npm run build
```

Expected: both commands exit 0, or any unrelated pre-existing failure is reported with its exact output.

- [ ] **Step 3: Perform a zero-background reader check**

Verify that a reader can answer for every experiment: why it was run, what was compared, what happened, what cannot be concluded, and why the next experiment followed.

- [ ] **Step 4: Review the final diff**

Run:

```powershell
git diff -- src/content/posts/ai4floorplan2.md
git status --short
```

Expected: only the requested article and the approved design/plan artifacts are attributable to this task; unrelated untracked user files remain untouched.

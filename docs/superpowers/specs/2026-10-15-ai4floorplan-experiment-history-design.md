# AI for Floorplan 实验探索历程扩写设计

日期：2026-10-15

## 目标

把 `src/content/posts/ai4floorplan2.md` 改写为一篇面向 floorplan 初学者的、按时间顺序组织的实验探索记录。文章要完整覆盖已经完成、得到阶段信号以及因保护门中止的实验，并清楚区分“动作空间有潜力”“离线 oracle 上界”“可部署收益”和“尚未得到结论”。

## 与上一篇的边界

- `ai4floorplan.md` 已经系统介绍 floorplan、HPWL、B\*-tree、官方评分、硬/软约束等基础知识；本文不重复这些内容。
- 本文只在新术语第一次出现时做就地解释，例如 `equal-compute`、`proposal recall`、`selector`、`abstention`。
- 开头链接上一篇，并提醒读者不熟悉基础概念时先阅读上一篇。

## 文章结构

正文按实验发生顺序组织为五个阶段：

1. **先找动作缺口**：Region Rebuild、one-step mutation、trajectory mutation。
2. **第一次真正让 LLM 上场**：Macro Agent v0、equal-compute、profile-only repeat。
3. **把动作做细并做 deterministic 收尾**：constructive B\*-tree-shaped ordering、proxy lookahead。
4. **从局部动作转向全局结构**：100-case proxy/oracle 复核、seed/reference 全局差距、直接坐标监督精修、SA/EA 自适应调度。
5. **找到新的可学习信号**：strong-incumbent reference oracle、learned-order proposer、label-free selector 准备与远程运行中止。

最后用一张总表总结每个实验的状态、关键数字与方向决策，并给出截至 2026-10-15 的阶段结论。

## 每节统一回答的问题

每个实验依次说明：

1. 为什么做；
2. 实验比较了什么；
3. 关键结果；
4. 结论可以覆盖到哪里；
5. 为什么下一步转向另一条路线。

## 证据口径

- 已有六节保留核心事实，但补齐对照、样本规模、关键数字和证据边界。
- trajectory 的早期部分运行信号与最终状态同时保留：多步出现过 gap，但 100-case full-pool 实验因成本停止，不能写成稳定通过。
- adaptive scheduler 与 Macro Agent v0 分开叙述；二者是不同实验线。
- learned-order 的约 8.06% 明确标为 offline exact best-of-16 proposal upper bound，不是在线 selector 的实际收益。
- selector 远程运行明确标记 `INCOMPLETE`，不能据此判断神经 selector 有效或无效。
- 不把 validation reference 当作已证明的全局最优解，也不把旧 16 个反复研究的 identities 当作 untouched test。

## 文风与可读性

- 延续现有第一人称研究日记风格，以直白类比解释实验设计和归因陷阱。
- 避免只堆数字；每张表前说明它回答的问题，表后用一句话解释读法。
- 英文术语首次出现时给出中文解释，后续保持用词一致。
- 保留失败实验，因为文章主题是如何逐步缩小问题，而不是包装一条全胜路线。

## 验收标准

- 时间线覆盖 2026-09-24 至 2026-10-15 的主要实验结论。
- 原博客已经写过的实验得到更完整的设置、数字和边界说明。
- 10 月 12–15 日已经完成或中止的实验全部进入正文。
- 基础概念不与 `ai4floorplan.md` 大段重复。
- Markdown frontmatter、图片引用和站点构建保持有效。
- 通过一次零背景读者检查：读者能够回答每个实验“为什么做、得到了什么、为什么转向”。

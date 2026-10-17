---
title: "个人对ai for floorplan的探索"
summary: "主要把走不通的方向列出来，进行阶段性summary，然后尝试提炼出真正可行的道路，这一篇是对contest的介绍，下一篇是我们的实验历程"
date: 2026-10-14
tags: ["agent", "eda", "floorplan", "model"]
category: "真实问题"
draft: false
---

# FloorSet Challenge：从 EDA 基础到 SoC Floorplanning 的完整问题建模

> 本文是 FloorSet 探索系列的第一部分，目标是面向熟悉机器学习、深度学习或组合优化，但尚未系统接触 EDA 的研究者，完整介绍 ICCAD 2026 CAD Contest Problem C：**The FloorSet Challenge — Data-Driven SoC Floorplanning**。本文重点是问题定义、数据、约束、评价指标、代码接口与研究难点；具体方法探索、实验过程和结果将在后续章节讨论。
>
> **版本说明：**本文依据 IntelLabs/FloorSet 仓库中的 `iccad2026contest/README.md`、比赛规则 PDF v10（2026-05-20 更新）和 `iccad2026_evaluate.py`。比赛规则可能与原始 FloorSet 论文及仓库顶层 README 不同；涉及竞赛得分时以下均以竞赛目录为准。

## 1. 这到底是一个什么比赛？

ICCAD（International Conference on Computer-Aided Design）是电子设计自动化领域的重要学术会议。其 CAD Contest 通常提供实际设计场景、数据和可运行的评估接口，要求参赛者解决特定的芯片设计自动化任务。

2026 年 Problem C 由 Intel 提供，题为 **The FloorSet Challenge: Data-Driven SoC Floorplanning**。给定一个芯片设计中若干模块的面积需求、模块之间的连接关系、外部引脚和布局约束，参赛者需要生成每个矩形模块的位置和形状，使最终布局**合法、紧凑、连接线短，并且计算时间合理**。

从优化角度看，它属于**带几何和组合约束的连续—离散混合优化问题**。布局变量本身是实数，但模块排列顺序、接触关系、邻接拓扑等具有离散结构；一个模块坐标的变化还可能影响多个其他模块的合法位置。

问题可以抽象为：

$$
\begin{aligned}
\text{Given}\quad &\mathcal I=(\mathcal B,\mathcal E_{bb},\mathcal E_{pb},\mathcal P,\mathcal C),\\
\text{Find}\quad &L=\{(x_i,y_i,w_i,h_i)\}_{i=1}^{N},\\
\text{to minimize}\quad &\operatorname{Cost}(L\mid\mathcal I),\\
\text{subject to}\quad &L\in\mathcal F_{\mathrm{hard}}.
\end{aligned}
$$

其中 $\mathcal B$ 表示模块集合，$\mathcal E_{bb}$ 是模块间连接，$\mathcal E_{pb}$ 是外部引脚与模块间连接，$\mathcal P$ 是引脚位置，$\mathcal C$ 是布局约束，$\mathcal F_{\mathrm{hard}}$ 是满足所有硬约束的布局集合。

**比赛不是要求模型输出一个图片。** 最终提交的结果是一组确定的几何量 `(x, y, width, height)`，官方评估器会根据这些数值计算线长、面积、约束违例与分数。因此，图像生成的视觉合理性不能代替精确几何合法性。

官方入口：

- 比赛主页：https://www.iccad-contest.org/Problems.html
- 代码仓库：https://github.com/IntelLabs/FloorSet
- 竞赛说明：https://github.com/IntelLabs/FloorSet/blob/main/iccad2026contest/README.md
- 竞赛实现：https://github.com/IntelLabs/FloorSet/tree/main/iccad2026contest

## 2. 先理解 EDA 与 Floorplanning

### 2.1 EDA、VLSI、SoC 是什么

**EDA（Electronic Design Automation，电子设计自动化）** 是使用算法与软件辅助完成芯片电路设计、验证和物理实现的一系列技术。与纯软件编译不同，芯片设计最终必须映射到真实的物理空间，受到面积、互连、功耗、时序、制造工艺等约束。

**VLSI（Very Large Scale Integration，超大规模集成电路）** 描述高度集成的大规模电路设计；**SoC（System on Chip，片上系统）** 则是将 CPU、GPU、缓存、内存控制器、I/O 等功能子系统整合在芯片上。SoC 设计中存在复杂的模块层次结构，物理布局的决策会影响布线距离、时序和芯片面积。

典型数字芯片实现流程包括：

1. RTL / 逻辑设计：描述和验证计算逻辑。
2. Logic Synthesis（逻辑综合）：将行为逻辑映射为门级网表。
3. Floorplanning（平面规划）：规划主要模块在芯片上的区域、形状和相对位置。
4. Placement（布局）：进一步安排标准单元等物理对象。
5. Clock Tree Synthesis（时钟树综合）：构造时钟分配网络。
6. Routing（布线）：将逻辑连接转化为实际的导线走线。
7. Sign-off（签核）：进行最终时序、功耗、物理规则等检查。

实际工业流程存在迭代，步骤顺序和内容也随设计工具与项目而变化。这里的主要区分是：**Floorplanning 决定较大功能模块的几何关系；标准单元 Placement 处理更细粒度的单元；Routing 才真正生成具体导线。**

### 2.2 Block / Macro / Partition

Floorplanning 的基本对象是一个 **block**（块、模块或 partition）。在本题中，一个 block 被抽象成轴对齐的矩形：

$$B_i=[x_i,x_i+w_i]\times[y_i,y_i+h_i].$$

$(x_i,y_i)$ 是左下角坐标，$w_i,h_i>0$ 是宽高，面积为 $A_i=w_ih_i$，中心为

$$c_i=\left(x_i+\frac{w_i}{2},\ y_i+\frac{h_i}{2}\right).$$

在真实 EDA 中，macro 经常指 SRAM 等较大的预设计模块，而 partition 也可能包含更大范围的逻辑。FloorSet 对这些结构作了简化，在数据中统一处理为具有面积、连线和限制条件的几何块。**数据里的矩形块不应理解为晶体管、标准单元或全部物理设计细节的真实还原。**

### 2.3 Net、Pin、Netlist 与 Connectivity

**Net（网络）** 是电路中的逻辑连接。**Pin（引脚/端子）** 是信号连接到模块或芯片边界的端点。**Netlist（网表）** 描述哪些对象连接到同一个 net。

FloorSet-Lite 的竞赛表示将连接简化成两类带权边：

- **Block-to-Block（B2B）**：模块 $i$ 与模块 $j$ 之间的连接。
- **Pin-to-Block（P2B）**：外部 pin $k$ 与模块 $i$ 之间的连接。

每条连接都有一个权重 $\omega$。权重较大的边在计算线长时贡献更大。因此，优化器不仅要减少几何距离，还要特别关注高权重连接。

这与图学习问题有直接对应关系：block 是节点，B2B 是节点之间的加权边，外部 pin 可以看成具有固定坐标的特殊终端。区别在于输出必须满足二维矩形 packing 的严格约束。

### 2.4 Floorplan 与 Packing 的关系

如果完全忽略连线，一个核心子问题就是 **Rectangle Packing（矩形装箱）**：给定若干面积或尺寸确定的矩形，在不重叠的前提下尽量紧凑地排布。

Floorplanning 比基础装箱问题复杂：除了避免重叠，还要控制不同模块的连接线长度、满足固定形状和预放置要求、处理相同形状模块、成组相邻的模块，以及要求处于整体布局边缘的模块。

另外，本比赛**移除了固定外框（fixed outline）限制**。也就是说，算法不需要将矩形装入一个预先给定宽高的芯片画布。相反，结果布局自身的外接矩形面积就是重要优化目标。这与原始 FloorSet 数据集中的 fixed-outline 问题不同。

### 2.5 Soft Block 与 Hard Block

这两个词容易与后文的**硬约束/软约束**混淆。

- **Soft block（可变形模块）**：目标面积指定，但宽高可以调整；本比赛未强制固定宽高比，只要求实现面积在目标值的 1% 范围内。
- **Fixed-shape block（固定形状模块）**：宽高必须保持数据给定的尺寸，但在没有其他限制时可以移动。
- **Preplaced block（预放置模块）**：位置和尺寸都固定，不允许改变。

这里的“soft block”说的是矩形的**形状可变**，不是说面积条件属于可忽略的软约束。恰恰相反，soft block 的 1% 面积容差是比赛**硬约束**。

## 3. FloorSet 数据集与比赛数据的区别

FloorSet 最初是一套面向大规模约束 floorplanning 研究的合成数据集，相关论文为 *FloorSet — a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs*（ICCAD 2024）。

完整 FloorSet 包含：

| 数据集 | 训练样本数 | 主要几何形式 | 与本次比赛关系 |
|---|---:|---|---|
| FloorSet-Prime | 1,000,000 | 可包含更复杂的多边形 partition | 原始数据集一部分，不是 Problem C 的指定训练集 |
| FloorSet-Lite | 1,000,000 | 矩形 partition | Problem C 指定训练数据 |
| 比赛 Validation | 100 | Lite 格式，21–120 blocks | 本地调参、比较和验证 |
| 比赛 Hidden Test | 100 | Lite 格式，21–120 blocks | 最终排名，不提供答案 |

原始两套训练集共计 200 万样本；但**不能把“原始 FloorSet 有 200 万条数据”直接写成“本次比赛有 200 万训练样本”**。对于 ICCAD 2026 Problem C，应以 100 万 Lite 训练样本为准。

这些样本是合成生成的，不等同于真实 Intel 产品的私有电路设计。训练样本提供输入及参考布局，可用于监督学习、模仿学习或学习启发式；也可以完全不训练模型，直接设计搜索、模拟退火、数学规划等求解器。

数据中的参考布局具有实际用途：它既可以作为监督标签，也提供线长和面积的**baseline（参考基准）**。但要特别区分：**数据提供一个已知参考解，并不意味着该解已被证明是全局最优解。** 比赛得分还刻意对优于 baseline 的指标做了截断，因此超越参考布局并不会在相应质量项上无限获得奖励。

### 3.1 从原始 FloorSet 到竞赛，哪些约束发生变化？

竞赛相对于原始数据设定主要放宽了三处：

| 项目 | ICCAD 2026 Problem C |
|---|---|
| 固定画布 fixed outline | 移除，输出布局的 bbox 面积直接计入代价 |
| Aspect ratio 宽高比 | 放宽，可以使用任意正宽高比 |
| 坐标类型 | 允许浮点坐标，不要求整数栅格 |

与此同时，部分容易混淆的约束仍然存在：固定形状、预放置是硬约束；boundary、grouping、MIB 是被评分的软约束。这里的 boundary 表示模块要接触**结果布局的外接矩形边缘**，并不表示存在一个预先固定的画布。

## 4. 数据格式：一个样本到底包含什么？

对于 AI 研究者，可以把每个 floorplan 样本视为一个包含图结构、几何条件、约束和参考答案的结构化对象。比赛训练加载器的典型返回值为：

```python
(
    area_target,
    b2b_conn,
    p2b_conn,
    pins_pos,
    constraints,
    tree_sol,
    fp_sol,
    metrics,
) = batch
```

这里 `tree_sol` 和 `fp_sol` 是 FloorSet-Lite 的两种参考答案表示。对于由竞赛优化器提交的结果，真正需要的是每个 block 的 `(x, y, w, h)`。

### 4.1 `area_target`：模块目标面积

通常形状为 `[batch_size, max_blocks]`，每个有效值是对应模块的目标面积 $a_i$。本比赛每例有效模块数量 $N\in[21,120]$。张量可能采用 padding，因此不能总是把张量长度理解成真实 block 数；官方示例通过 `area_target != -1` 计算有效 block 数。

例如：

```python
area_target = [100., 64., 144., -1., -1.]
block_count = sum(a != -1 for a in area_target)  # 3
```

这只是展示 padding 机制的简化样例，不是比赛的完整实例。模块 $0$ 目标面积为 100，可以选择 $(w,h)=(10,10)$ 或 $(20,5)$，但其他约束可能进一步限制选择。

### 4.2 `b2b_connectivity`：模块间连接表

比赛评测代码将 B2B 的一条边解释为

```text
[block_i, block_j, weight]
```

例如：

```python
b2b_conn = [
    [0, 1, 2.0],
    [1, 2, 5.0],
]
```

表示 block 0–1 的线长乘以 2，block 1–2 的线长乘以 5。实际数据可能是 padding 后的张量，评估器通过 `edge[0] == -1` 跳过无效边。

严格来说，这是**稀疏加权边表（edge list）**，不必先变换成 $N\times N$ 邻接矩阵。用于 GNN 时可以变换为 `edge_index` 与 `edge_attr`；用于传统求解器时，直接遍历边表即可。

### 4.3 `p2b_connectivity` 和 `pins_pos`

P2B 一条边的格式是：

```text
[pin_index, block_index, weight]
```

`pins_pos` 为 pin 的坐标列表，每行 `[x_pin, y_pin]`。假设：

```python
pins_pos = [[0.0, 10.0], [40.0, 20.0]]
p2b_conn = [[0, 2, 3.0]]
```

含义是外部 pin 0 与 block 2 相连，权重为 3。优化器移动 block 2 时，该连接的代价将发生变化；但 pin 坐标由数据给定。

**注意：**这里的 pin-to-block 线长是简化几何代理，并不意味着工具真的执行了芯片详细布线。

### 4.4 `placement_constraints`：五列约束编码

每个 block 具有 5 个约束字段，行结构是：

```text
[fixed_flag, preplaced_flag, mib_group_id, cluster_group_id, boundary_mask]
```

| 列索引 | 字段 | 含义 |
|---:|---|---|
| 0 | `fixed` | 1 表示宽高固定 |
| 1 | `preplaced` | 1 表示位置与宽高固定 |
| 2 | `mib_id` | 大于 0 表示属于同形状模块组 |
| 3 | `cluster_id` | 大于 0 表示属于必须保持连通的模块组 |
| 4 | `boundary_mask` | 二进制位标志，指定必须接触哪些外接矩形边 |

例如，`[0, 0, 2, 3, 5]` 可以解读为：当前模块不属于固定尺寸和预放置块，属于 MIB 组 2、cluster 组 3，并且要求同时接触左边界和上边界。它是说明编码语义的示例，不代表这个组合一定出现在某条实际样本里。

#### Boundary mask 的位编码

| 边界 | 二进制对应位 | 数值 |
|---|---:|---:|
| LEFT | bit 0 | 1 |
| RIGHT | bit 1 | 2 |
| TOP | bit 2 | 4 |
| BOTTOM | bit 3 | 8 |

通过按位或即可表示多个条件。例如 $5=1+4$ 表示左上角，$6=2+4$ 表示右上角，$9=1+8$ 表示左下角，$10=2+8$ 表示右下角。

```python
def decode_boundary(mask: int):
    return {
        "left":   bool(mask & 1),
        "right":  bool(mask & 2),
        "top":    bool(mask & 4),
        "bottom": bool(mask & 8),
    }
```

注意 corner 要求两个对应的边**都**接触，不是“靠近角落”即可。

### 4.5 `fp_sol`、`tree_sol`：标签与结构化布局

对于 Lite 数据，参考 floorplan 可表示成每个 block 的四元组 `[w,h,x,y]`。**不要与参赛优化器输出格式 `(x,y,w,h)` 混用**，否则宽高和坐标会被错误解释。

训练 batch 同时有 `tree_sol`，对应 B\*-tree 结构表示。B\*-tree 是一种表达矩形相对布局关系的树结构，后面介绍模拟退火 baseline 时再展开。对普通机器学习训练而言，可以选择将 `fp_sol` 用于监督位置/形状预测；对结构预测模型而言，可以考虑学习 `tree_sol` 的拓扑信息。

固定形状或预放置块的目标宽高、位置也要从提供的参考布局中取得；仅有五列约束标志，不足以恢复目标数值。

### 4.6 `metrics`：参考解指标

仓库对数据集 `metrics` 给出的字段定义为：

```text
[
    area,
    num_pins,
    num_total_nets,
    num_b2b_nets,
    num_p2b_nets,
    num_hardconstraints,
    b2b_weighted_wl,
    p2b_weighted_wl,
]
```

其中 `area` 是参考布局外接矩形面积，最后两个是参考布局的两部分加权线长。总参考线长可写为

$$WL_0=WL_{bb,0}+WL_{pb,0}.$$

不要把 `area_target` 每个 block 的面积之和与 `metrics[0]` 混为一谈：前者是模块占据的面积总和，后者是**整个布局 bounding box 的面积**。布局中的空白间隙会使二者不同。

> **编码提醒：**不同数据加载器、batch size 和 padding 可能改变外层张量维度。实际训练时应打印 shape、dtype、有效 block 个数及无效边数量，再构建模型。本文的列表样例只是结构解释，不是假定真实文件都是 Python list。

## 5. 优化变量与三类最基本的几何指标

设求解结果为

$$L=\{(x_i,y_i,w_i,h_i)\}_{i=1}^{N}.$$

布局算法要决定每个 block 的位置与宽高，同时使各种连接尽量短，布局尽量紧凑。

### 5.1 Weighted HPWL：连线长度

**HPWL（Half-Perimeter Wirelength）** 是 EDA 中常见的线长代理指标。对于具有多个端点的 net，其定义通常为

$$HPWL(e)=(\max_{p\in e}x_p-\min_{p\in e}x_p)+(\max_{p\in e}y_p-\min_{p\in e}y_p).$$

它等于所有端点的轴对齐外接矩形半周长。在本比赛的 B2B/P2B **二端点加权边**表示中，HPWL 退化为端点之间的曼哈顿距离：

$$d_1(p,q)=|x_p-x_q|+|y_p-y_q|.$$

官方实现用 block **中心点**表示模块连接位置，因此

$$WL_{bb}=\sum_{(i,j,\omega)\in\mathcal E_{bb}}\omega\left(|c_i^x-c_j^x|+|c_i^y-c_j^y|\right),$$

$$WL_{pb}=\sum_{(k,i,\omega)\in\mathcal E_{pb}}\omega\left(|p_k^x-c_i^x|+|p_k^y-c_i^y|\right),$$

$$WL=WL_{bb}+WL_{pb}.$$

例子：两个 block 中心分别是 $(3,4)$ 和 $(12,10)$，连接权重为 $2$，这一条边贡献

$$2\bigl(|12-3|+|10-4|\bigr)=30.$$

这并不是 detailed routing 的实际金属线长度，也没有直接模拟障碍物绕线、拥塞、寄生参数或时序。

### 5.2 Bounding Box Area：整体面积

设所有模块的最小外接矩形边界为：

$$x_{\min}=\min_i x_i,\quad x_{\max}=\max_i(x_i+w_i),$$

$$y_{\min}=\min_i y_i,\quad y_{\max}=\max_i(y_i+h_i).$$

那么布局面积是

$$A_{bbox}=(x_{\max}-x_{\min})(y_{\max}-y_{\min}).$$

这个指标会间接惩罚模块之间的无效空隙。由于没有固定 outline，算法既可以调整模块间的相对位置，也可能改变布局整体长宽比。

### 5.3 Overlap：模块重叠

设两个矩形为 $i,j$，其水平方向的交叠长度和垂直方向的交叠长度分别为

$$o^x_{ij}=\max\bigl(0,\min(x_i+w_i,x_j+w_j)-\max(x_i,x_j)\bigr),$$

$$o^y_{ij}=\max\bigl(0,\min(y_i+h_i,y_j+h_j)-\max(y_i,y_j)\bigr).$$

交叠面积为 $O_{ij}=o^x_{ij}o^y_{ij}$。当两个方向都有正交叠长度时，模块内部相交；仅共享一条边或一个顶点通常不算非法重叠。官方 `check_overlap` 对两个方向的交叠长度使用 `1e-6` 数值阈值。

**避免重叠不是单纯的线长优化。** 一个布局可能令所有 block 中心点靠得非常近，从而得到极低的 HPWL，但如果矩形重叠，它仍然是无效解。

## 6. 比赛约束：硬约束与软约束

这一部分尤其重要。最初的 FloorSet、原始论文和 2026 比赛早期版本对部分约束的分类可能不同。以下以竞赛 v10 和当前评测实现为准。

### 6.1 硬约束：违反任意一条即判不可行

**（1）No Overlap。** 任意两个不同 block 的内部不能有有效面积的重叠。

**（2）Soft-block Area Tolerance。** 对不属于 fixed/preplaced 的普通 soft block，要求

$$\frac{|w_ih_i-a_i|}{a_i}\le 0.01.$$

因此面积不能为了追求更低线长而任意缩小。

**（3）Fixed Shape。** 带 fixed 标志的模块宽高必须等于给定尺寸：

$$w_i=w_i^*,\quad h_i=h_i^*.$$

位置可以变化，除非还有其他约束。竞赛实现对这些尺寸使用约 $10^{-4}$ 的数值容差，不应把这种浮点容差误认为可以主动调整固定形状。

**（4）Preplaced。** 带 preplaced 标志的模块必须同时满足

$$(x_i,y_i,w_i,h_i)=(x_i^*,y_i^*,w_i^*,h_i^*).$$

这里 $(\cdot)^*$ 表示数据中的参考规定值。它不允许被优化器随意平移。

只要发生任意一种硬约束违例，该测试样本就被判 `infeasible`，`Cost = 10`。官方评估器在实现上用独立函数检查重叠、soft-block 面积和固定/预放置尺寸位置。

### 6.2 软约束一：Boundary

带 boundary 条件的 block 必须触碰解的 bbox 指定的边（或角）。例如 `boundary_mask=5` 要求同时满足

$$x_i=x_{\min},\qquad y_i+h_i=y_{\max}.$$

这里的等号在实际代码中使用适当几何容差判定。对每个带 boundary 条件的模块，**任何一个要求的边不满足时，该 block 计一次 boundary violation**。

### 6.3 软约束二：Grouping / Cluster

一组带同一 `cluster_id` 的 block 要形成一个通过**共享边**相连的几何整体，不是只靠角接触，也不是要求它们在空间上距离近。

可以构造一张只含该组模块的接触图：模块为节点；只有两矩形沿非零长度的边接触时才连边。如果该图有 $c_g$ 个连通分量，则这一组的违例数量为

$$V_{group,g}=c_g-1.$$

完全连通时 $c_g=1$，违例为 0；三个模块全部彼此隔离时 $c_g=3$，违例为 2。这将几何检查转化为图连通性检查。

### 6.4 软约束三：MIB（Multi-Instantiation Blocks）

MIB 表示多个模块是同一个 master partition 的实例，要求相同组内模块具有**相同的宽高**。注意其要求比“面积一样”更强：

$$\forall i,j\in M_g:\quad w_i=w_j,\ h_i=h_j.$$

例如 $(w,h)=(5,10)$ 和 $(10,5)$ 面积相同，却不满足相同尺寸的条件。

若某组共有 $s_g$ 种不同的尺寸对，那么组违例数量按

$$V_{mib,g}=s_g-1$$

计算。固定尺寸、预放置和 MIB 等约束相互叠加时，求解器需要先处理它们是否兼容，再决定可调整的参数。

### 6.5 归一化软约束违例

把三类软违例相加：

$$V_{soft}=V_{boundary}+V_{grouping}+V_{mib}.$$

为避免不同实例中约束总数不同，比赛使用

$$V_{rel}=\frac{V_{soft}}{N_{soft}},$$

其中

$$N_{soft}=|B_{boundary}|+\sum_g(|G_g|-1)+\sum_q(|M_q|-1).$$

这里 $|B_{boundary}|$ 是带边界条件的 block 数量，$G_g$ 和 $M_q$ 分别是 cluster 与 MIB 分组。对于没有软约束的特殊情形，评测实现需要避免除零；概念上其 $V_{rel}=0$。正常归一化下 $V_{rel}\in[0,1]$。

这也意味着**不能把五列 constraint 都作为连续 soft loss 一律相加**：fixed 和 preplaced 已经升级为硬约束，并不进入 $N_{soft}$ 分母。

## 7. 官方 Score：完整公式与含义

这是本题的核心。设某个测试样本的最终线长、bbox 面积和参考指标分别是 $WL, A$ 和 $WL_0,A_0$。定义相对差距：

$$g_{WL}=\frac{WL-WL_0}{\max(WL_0,10^{-6})},\qquad g_A=\frac{A-A_0}{\max(A_0,10^{-6})}.$$

官方会对负的 gap 截断：

$$g_{WL}^+=\max(0,g_{WL}),\qquad g_A^+=\max(0,g_A).$$

因此如果算法比参考解更好，**该项只被记为零差距**，不会因为优于 baseline 而在质量因子中获得负奖励。

若布局合法，定义

$$Q=1+0.5(g_{WL}^++g_A^+),$$

$$P=\exp(2V_{rel}),$$

$$T=\max(0.7,R^{0.3}),$$

其中 $R$ 为求解时间相对同一个测试样本上所有参赛方案**运行时间中位数**的比值，即

$$R=\frac{t_{solver}}{\operatorname{Median}_{submissions}(t_{\mathrm{same\ case}})}.$$

最终单例代价是

$$
\boxed{
C=\begin{cases}
10,&\text{if any hard constraint is violated},\\[2mm]
\min\left(Q\,P\,T,\ 10-10^{-6}\right),&\text{otherwise}.
\end{cases}
}
$$

**分数越低越好。** 其中：

- $Q$ 惩罚线长和面积相对于参考解的退化，系数 $0.5$；
- $P$ 指数惩罚 grouping、MIB、boundary 的相对违例，系数 $2$；
- $T$ 考虑相对运行时间，指数 $0.3$，最低为 $0.7$，因此加速收益被限制在 30%；
- `10 - 1e-6` 是对合法解的封顶，确保**无论质量多差，合法解的单例分数仍严格低于非法解的 10 分**。

在实现中，runtime factor 还经过最小值数值保护；这不是允许耗时为零或出现非法运行时间的意思。

### 7.1 几个直观的数值例子

假设方案合法且运行时间与中位数相同，即 $R=1$：

| 情况 | $g_{WL}^+$ | $g_A^+$ | $V_{rel}$ | Cost（约） |
|---|---:|---:|---:|---:|
| 达到参考解，软约束全满足 | 0 | 0 | 0 | 1.000 |
| 线长恶化 20%，面积不变 | 0.2 | 0 | 0 | 1.100 |
| 线长和面积均恶化 20% | 0.2 | 0.2 | 0 | 1.200 |
| 指标达到参考解，但相对违例 0.1 | 0 | 0 | 0.1 | $e^{0.2}\approx1.221$ |
| 任意一个硬约束不满足 | — | — | — | 10.000 |

同一个合法解若明显快于其他参赛方案，$T$ 可以降到 $0.7$，理论最低单例 cost 为 0.7。反过来运行慢可能带来额外惩罚。

需要特别强调：**本地 100 条 validation 评估把 `RuntimeFactor` 固定为 1.0**，因为本地不知道其他队伍在每一道题上的运行时间中位数。因此本地分数主要衡量几何质量和约束违例；不能把本地分数直接当成最终官方排行榜分数。

### 7.2 100 个测试样本如何汇总？

最终总分不是简单的 100 个 cost 平均数。设第 $i$ 个测试样本有 $n_i$ 个 block，则

$$
\boxed{\operatorname{TotalScore}=\frac{\sum_{i=1}^{100}C_i\exp(n_i/12)}{\sum_{i=1}^{100}\exp(n_i/12)}}.
$$

即**block 数越多，权重越高**。官方 v10 把旧版本中的 $e^{n_i}$ 改为 $e^{n_i/12}$，避免最大的测试实例几乎独占所有权重，同时仍然鼓励算法处理更复杂的布局。

官方 README 给出的分段权重近似为：116–120 blocks 占 34%，111–115 占 22%，106–110 占 15%，101–105 占 10%；换言之，较大实例的性能会显著影响最终成绩。

这在实验设计中有两个直接后果：一是只报告 100 题的平均线长或平均面积不足以反映比赛成绩；二是对大规模案例出现少量非法解可能比在小案例中获得若干温和改进更加致命。因此，应同时报告 overall score、feasibility rate、不同规模区间的分数、软约束违例和运行时间。

## 8. 官方 Training Loss 为什么不等于 Score？

如果打算使用深度学习，需要把**可微训练目标**与**最终离散评测规则**严格分开。

官方仓库在 `training_example.py` 中演示如何通过 `compute_training_loss_differentiable(...)` 训练模型：输入预测矩形、连接数据、pin 坐标、目标面积和参考指标，得到能够执行 `backward()` 的 loss。

它是训练用的**代理目标（proxy objective）**，涉及：

1. 相对参考布局的线长差距；
2. 相对参考布局的 bbox 面积差距；
3. 可微的重叠惩罚；
4. 可微的面积容差违例惩罚。

为了说明设计思路，可以把这样一类 proxy 写为

$$\mathcal L_{proxy}\sim \mathcal L_{quality}(WL,A;WL_0,A_0)+\lambda_o\mathcal L_{overlap}+\lambda_a\mathcal L_{area},$$

其中一个可微的面积偏差 surrogate 可以定义为

$$\mathcal L_{area}=\sum_i\operatorname{ReLU}\left(\frac{|w_ih_i-a_i|}{a_i}-0.01\right),$$

重叠 surrogate 可以使用上文的交叠面积之和

$$\mathcal L_{overlap}=\sum_{i<j}O_{ij}.$$

**注意：上面两项是用于解释可微近似的示例表达式，并非声称逐字复刻官方源码内部的权重和数值处理。** 要完全复现实验，应直接调用仓库函数并核对所固定的 commit。

最关键的差异是：官方 README 明确说明训练用可微 loss **没有完整纳入 fixed、preplaced、MIB、grouping 和 boundary 约束**，也没有官方最终排名使用的跨队伍运行时间因子。换言之，某个模型即使优化 proxy loss 收敛得很好，生成结果仍可能因硬约束而直接得到 10 分。

对于机器学习方法，通常还需要额外考虑以下技术路线：使用满足约束的结构化输出表示、把部分约束硬编码到参数化中、预测后进行 legalization（合法化），或将神经网络用于初始化及候选动作排序，再交给传统几何搜索模块处理可行性。这里先介绍问题，不提前判断哪条路线一定更好。

## 9. B\*-tree 与模拟退火：为什么官方 baseline 不是直接预测坐标？

官方 `optimizer_template.py` 提供了 **B\*-tree + Simulated Annealing（模拟退火，SA）** 风格的 baseline。理解它可以帮助区分“布局表示”和“优化算法”。

### 9.1 直接坐标表示的难点

如果让模型直接预测 $N\times4$ 个连续数值，必须同时确保面积满足、固定块不变、块不重叠，且尽可能满足各种组约束。独立预测会很容易产生相互覆盖的矩形。

### 9.2 B\*-tree 是什么

B\*-tree 用有序二叉树表达模块之间的相对位置关系。经典 B\*-tree 的 packing 规则通常是：左孩子对应位于父模块右侧的模块，右孩子对应从特定水平关系开始排布的模块；使用 contour（轮廓线）数据结构将树变成实际布局，并消除构造过程中的不合法重叠。

例如，改变树中的父子关系或交换两个节点，实际上改变的是模块相对布局结构；之后重新执行 packing，才得到最终 $(x,y)$ 坐标。其重要意义不在于某棵树能直接给出最优布局，而在于它把一部分难处理的几何合法性问题转移到了**布局表示和确定性的解码过程**中。

需要注意：非重叠 packing 不意味着 fixed/preplaced、cluster、MIB、boundary 自动满足，这些仍要额外处理。

### 9.3 SA 如何搜索树空间

SA 是一种随机局部搜索算法。维护当前候选 $s$，对它进行扰动得到 $s'$，并计算某个用于搜索的能量函数 $E(s)$。令 $\Delta E=E(s')-E(s)$，则经典接受概率为

$$P(\mathrm{accept})=\begin{cases}1,&\Delta E\le0,\\e^{-\Delta E/T},&\Delta E>0.\end{cases}$$

$T$ 是温度参数，通常随迭代逐步降低。高温阶段有概率接受更差的候选，以减少陷入局部极值的风险；低温阶段则倾向于保留更优解。

在 floorplanning 中，候选动作可能包括交换 block、改变拓扑或移动节点等，随后重新 packing、计算线长、面积及约束代价。值得区分的是：**搜索阶段的能量函数可以与比赛最终 Score 不同**，例如搜索时使用额外的平滑惩罚来引导逐步改善；最终比较仍应运行官方 evaluator。

因此，B\*-tree 是一种结构化解表示，SA 是在该表示上搜索的策略，两者不是同一个概念。

## 10. 从数据到得分：最小可复现的代码流程

### 10.1 准备环境

```bash
git clone https://github.com/IntelLabs/FloorSet.git
cd FloorSet
python -m venv venv
source venv/bin/activate     # Linux / macOS
pip install -r iccad2026contest/requirements.txt
cd iccad2026contest
```

Windows 环境可使用适合自身 shell 的虚拟环境激活命令。官方提供的 dataloader 可以自动获取数据，也可以手动将训练和验证文件放到仓库根目录的 `LiteTensorData/`、`LiteTensorDataTest/`。

### 10.2 探查数据结构

```python
from iccad2026_evaluate import get_training_dataloader

loader = get_training_dataloader(batch_size=1, num_samples=10)
batch = next(iter(loader))

names = [
    'area_target', 'b2b_conn', 'p2b_conn', 'pins_pos',
    'constraints', 'tree_sol', 'fp_sol', 'metrics'
]

for name, value in zip(names, batch):
    print(name, getattr(value, 'shape', type(value)))

area_target = batch[0].squeeze(0)
block_count = int((area_target != -1).sum().item())
print('valid block_count =', block_count)
```

这段代码用于查看实际数据维度，不需要提前决定采用 CNN、GNN、Transformer 还是传统算法。对于变量长度数据，应特别检查 padding 和有效边 mask。

### 10.3 使用官方模板实现求解器

```bash
cp optimizer_template.py my_optimizer.py
```

根据官方约定，参赛求解器的核心接口是：

```python
def solve(self, block_count, area_targets,
          b2b_connectivity, p2b_connectivity,
          pins_pos, constraints):
    # 在这里执行布局算法
    # 返回的每项顺序是 x, y, width, height
    positions = []
    ...
    return positions
```

这里的省略号只表示需要用户实现的部分，不是可以直接运行的合法求解器。不要将所有模块的 `(x,y)` 都设为零，那会制造大规模重叠。推荐先复制官方可运行模板，在其基础上逐步替换 `solve()` 的内部算法，以免接口不匹配影响评估。

### 10.4 本地测试与保存解

```bash
# 检查提交接口
python iccad2026_evaluate.py --validate my_optimizer.py

# 单案例调试
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0

# 100 个 validation 测试
python iccad2026_evaluate.py --evaluate my_optimizer.py

# 保存并重新评分：避免重复运行搜索算法
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
python iccad2026_evaluate.py --score my_optimizer_solutions.json

# 查看数据与可视化
python iccad2026_evaluate.py --training
python iccad2026_evaluate.py --visualize --test-id 0
```

对于随机优化器，建议固定 seed，并保存原始 positions、每个 case 的 metrics、求解耗时和算法配置。只记录一个 Total Score 很难定位算法到底在哪类实例和哪种约束上失败。

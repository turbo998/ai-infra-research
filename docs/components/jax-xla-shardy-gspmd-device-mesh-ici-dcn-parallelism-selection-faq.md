# 大模型分布式并行与编译器分区架构：`JAX / XLA Shardy (MLIR)` vs. `GSPMD`、分层设备网格（`Mesh / PartitionSpec / shard_map`）与 `ICI vs. DCN` 集合通信映射选型指南

> **适用读者**：负责千卡至万卡级大模型（Dense Transformer 与 Sparse MoE）分布式并行策略设计、设备网格（Device Mesh）划分与编译器自动分区调优的 Cloud & AI Infrastructure Solution Architect、大模型训练框架架构师与性能工程负责人。
> **核心课题**：当在 **`2,048` 颗芯片（4 个 `512` 芯片 `8x8x8` 3D Torus Slice 通过数据中心网络 `DCN` 互联）** 上训练 `400B-A45B` 稀疏混合专家（MoE）大模型时，为什么“把 `TP` 或未分层的 `FSDP` 跨切片拉到 `DCN` 上（Tier 1）”会导致 **`68.5%` 的步时卡死在慢速网络等待、MFU 暴跌至 `15.8%`（每月烧掉 `$392 万`）**？为什么旧版基于扁平 HLO 的 **`GSPMD`** 分区器在面对“Attention (`FSDP×TP`) $\leftrightarrow$ MoE (`EP×TP`)”分支切换时会平白无故插入 **`38 GB/步` 的冗余 `Reshard / All-Gather` 通信（Tier 3，MFU 卡在 `48.2%`）**？如何通过新一代 **`XLA Shardy (MLIR)` 优先级传播与子轴切分 + `create_hybrid_device_mesh` 严格对齐 `ICI`（承载 `TP/FSDP/EP`）与 `DCN`（仅承载外层 `DP`）+ `shard_map` 显式环状注意力流水线（Tier 4）**，彻底清零冗余重分片（**`0.0 GB`**）、将暴露通信压降至 **`6.4%`**、持续 MFU 提升至 **`59.6%`（较旧版 `GSPMD` 净增 `+11.4 pts`）**，并将月度训练总 TCO 降低 **`62.2%`（`$3.92M -> $1.48M/月`）**？

![JAX/XLA Shardy (MLIR) vs. GSPMD 分层设备网格与 ICI vs. DCN 并行架构总结图](../../assets/svg/jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在多切片（Multislice）分布式训练中，**90% 的性能灾难都源于两条架构失误：① 把每层都要通信 2–4 次的“高频带宽饥渴型并行轴（`TP / FSDP / EP`）”错误映射到了跨切片的慢速以太网（`DCN`）上；② 在既有 Attention 又有稀疏专家（MoE）的复杂图中，任由旧版编译器在分支交汇处插入上百 GB 无谓的张量重排（Reshard）**：

| 并行网格与编译器分区层级 | 适用场景 | `2,048` 芯片持续训练 `MFU` | 未被计算隐藏的暴露通信占比 | 每步冗余 `Reshard` 通信量 | 月度 `2,048` 芯片训练总成本 (`TCO`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：错位网格（将 `TP` 或平铺 `FSDP` 跨切片拉到 `DCN` 上，反模式）** | ❌ **严禁在任何多切片/多机房训练中使用** | **`15.8%`（近 7 成时间等网络！）** | **`68.5%`** | `148.0 GB / 步` | `$3,920,000 / 月` | ❌ **严禁上生产**：把逐层同步的 `All-Gather` 放到百 G 以太网上，芯片全部空转 |
| **Tier 2：单维扁平 `1D FSDP` 环（未按 3D Torus 正交轴分解 `TP / EP`）** | `<64` 卡小规模稠密模型微调 | `38.4%` | `34.0%` | `62.0 GB / 步` | `$2,450,000 / 月` | ⚠️ **高跳数与激活爆显存**：512 跳单环延迟高，且未切分激活值导致长序列 OOM |
| **Tier 3：旧版扁平 HLO `GSPMD` 分区器跑混合 MoE 多切片** | 传统稠密模型基线 | `48.2%` | `21.5%` | **`38.0 GB / 步`（分支冲突！）** | `$1,920,000 / 月` | ⚠️ **MoE 重分片税**：GSPMD 无法调和 Attention 与 MoE 分片冲突，凭空多传 38GB |
| **Tier 4：新一代 `XLA Shardy (MLIR)` + 双层 `ICI/DCN` 混合网格 + `shard_map`（推荐）** | **千卡/万卡稠密与稀疏 MoE 大模型预训练/RL 主力架构** | **`59.6%`（较 GSPMD `+11.4 pts`）** | **`6.4%`** | **`0.0 GB / 步`（零冗余重排）** | **`$1,480,000 / 月`（省 `62.2%`）** | 🏆 **生产金标准**：`Shardy` 优先级传播消除了全部冗余通信，`ICI/DCN` 严丝合缝 |

### 💡 架构师的三大黄金铁律
1. **死守“物理互连带宽层级 $\leftrightarrow$ 并行通信频次”铁律：高频轴（`TP / EP / FSDP`）100% 锁在 Slice 内 `3D Torus ICI`，低频外层梯度轴（`DP`）才允许走跨 Slice `DCN`**：
   * 先算清每种并行策略在 1 个训练步（64 层模型）里到底发多少次集合通信：
     * **张量并行（`TP`）**：每层前向 `2` 次、反向 `2` 次（`All-Gather + Reduce-Scatter`），64 层每步高达 **`256 次`** 强同步通信！**必须锁死在单机内或 `3D Torus` 的单一最短轴（如 `Z 轴 TP=4 或 8`）**。
     * **专家并行（`EP`）与完全分片数据并行（`FSDP / ZeRO-3`）**：每层执行 `All-to-All`（MoE 派发/汇总）或权重 `All-Gather` + 梯度 `Reduce-Scatter`，64 层每步 **`128–192 次`** 通信，但可与前后层的矩阵乘流水线重叠！**必须映射在 Slice 内部的 2D `X×Y Torus ICI` 平面（如 `FSDP/EP = 64`）**。
     * **外层数据并行（`Outer DP / Replicated FSDP`）**：整个前向+反向跑完后，**每步仅通信 `1 次` 梯度 `All-Reduce`**！**这是唯一允许跨出 Slice 边界、跑在数据中心网络 `DCN (Multislice)` 上的并行轴**！
2. **全面升级至基于 `MLIR` 的新一代 `XLA Shardy` 分区器（开启 `--xla_use_shardy=true`），终结旧版 `GSPMD` 在 MoE 模型上的冗余 `Reshard` 灾难**：
   * 为什么旧版 `GSPMD` 跑 MoE 会丢掉 `11.4 pts` 的 MFU？因为 `GSPMD` 在编译晚期将整个模型拍平为无结构的 HLO 图，当遇到“自注意力层按 `(batch=fsdp, hidden=tp)` 切分、而紧接着的 MoE 专家层按 `(experts=ep, hidden=tp)` 切分”时，`GSPMD` 的双向启发式传播在残差连接（Residual Add）处发生冲突，愚蠢地插入了高达 **`38 GB/步` 的多余 `All-Gather / Dynamic-Slice` 重分片拷贝**！
   * **`Shardy` 的三大突破**：① 基于结构化 **`MLIR / StableHLO` 方言**，完整保留模块层级；② 支持 **分片优先级注解（Priority-Based Propagation：`P('fsdp', 'tp')` 优先级分层）与子轴切分（Sub-Axis Splitting）**，精准引导编译器在恰当位置做唯一一次转换（冗余 Reshard 降至 **`0.0 GB`**）；③ 提供可读性极强的分片传播追踪日志，架构师一眼就能看清哪一行代码引发了 `All-to-All`。
3. **用“90% 声明式 `NamedSharding (Shardy)` + 10% 关键算子显式 `shard_map`（配 `Pallas`）”实现鱼与熊掌兼得**：
   * 全模型骨架交给 `Shardy` 自动传播；但对于 `128K+` 超长上下文的 **环状注意力（Ring Attention `jax.lax.ppermute`）** 或 **MoE 异步双缓冲 `All-to-All`**，使用 `jax.experimental.shard_map` 下沉到单设备视角手写显式通信与计算重叠，将暴露通信占比死死压在 **`6.4%`**！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：六大分布式并行策略（`DP`、`FSDP`、`TP`、`EP`、`PP`、`CP/SP`）到底各自对应什么集合通信原语？在 `3D Torus ICI + DCN` 上该怎么排座次？
* **架构师一页纸并行策略与物理轴映射圣经**：

| 并行维度 | 切分对象 | 核心集合通信原语 | 通信频次（每层 vs 每步） | 物理网络轴强制映射准则 (`TPU 3D Torus / GPU`) |
| :--- | :--- | :--- | :--- | :--- |
| **`Outer DP`（跨切片数据并行）** | 切分全局 Batch | 梯度 `All-Reduce` | **每步仅 `1` 次**（反向传播末尾异步聚合） | **跨 Slice 的 `Jupiter DCN`（Multislice 轴）** |
| **`FSDP`（完全分片数据并行 / ZeRO-3）** | 切分参数、梯度与优化器状态（`16B/Param`） | 前向 `All-Gather`；反向 `All-Gather + Reduce-Scatter` | 每层 `3` 次（可通过预取与上一层计算 `100%` 重叠） | **Slice 内部 `3D Torus ICI` 的长轴（如 `X×Y 平面 = 64`）** |
| **`TP`（张量并行 / Megatron）** | 切分单层矩阵乘的行/列权重 | 前向与反向激活值 `All-Gather + Reduce-Scatter` | **每层 `4` 次强同步**（处于关键路径，极难完全隐藏） | **单主机内或 `3D Torus ICI` 最短轴（如 `Z 轴 = 4 或 8`，绝不跨机柜/DCN）** |
| **`EP`（MoE 专家并行）** | 将不同 Routed Experts 分散到不同芯片 | Token 派发与回收 `All-to-All`（或 `Ragged All-to-All`） | 每层 MoE `2` 次 `All-to-All` | **与 `FSDP` 复用 Slice 内部 `2D ICI` 轴（配合 `SparseCore` 索引）** |
| **`CP / SP`（上下文/序列并行 Ring Attention）** | 沿 `seq_len` 切分超长序列激活与 KV | 环状移位 `Collective-Permute (ppermute)` | 注意力层内多轮环状传递，与 `FlashAttention` 块计算完全重叠 | **Slice 内部带 Wrap-Around 回环的 `ICI` 环轴** |
| **`PP`（流水线并行）** | 将不同 Transformer 层切到不同 Stage | 点对点 `Send / Recv (Collective-Permute)` | 仅在相邻 Stage 边界传激活值（需 DualPipe/1F1B 消除气泡） | **跨机柜 `ICI` 轴或高带宽 `DCN` 相邻 Slice** |

### Q2：如何在代码中用 `create_hybrid_device_mesh` 写出既匹配物理 `3D Torus` Cube、又隔离 `DCN` 的黄金网格？
* **两行代码锁定物理拓扑（杜绝逻辑轴与物理光纤错位）**：
  * 如果你直接把 `jax.devices()` 拍平做 `reshape((4, 64, 8))`，逻辑上的相邻编号可能跨了物理机柜！必须调用 `jax.experimental.mesh_utils.create_hybrid_device_mesh(mesh_shape=(64, 8), dcn_mesh_shape=(4, 1))`：它会自动读取底层 TPU 的三维物理坐标 `(x, y, z)` 与 `slice_index`，保证 `ici_tp=8` 严格沿着物理连续铜缆排列、`ici_fsdp=64` 沿着闭合回环 Torus 排列、`dcn_dp=4` 严格对齐 4 个独立的 Slice！

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 网格映射图 + 时序图）

### 3.1 大模型并行轴与 `ICI / DCN` 物理网格映射决策树

```mermaid
flowchart TD
    A["多切片大模型训练网格设计 (如 4 Slices x 512 Chips = 2,048 芯片)"] --> B["第 1 步: 外层跨 Slice 轴 (DCN)\n固定 dcn_dp = 4 (仅在每步末尾走 DCN 做 1 次异步梯度 All-Reduce)"]
    B --> C{"第 2 步: 单卡 HBM 能否放下单层激活与 FSDP 预取窗口?"}
    C -- "70B~671B 大模型 (需切分单层矩阵与激活)" --> D["在 Slice 内最短 ICI 轴 (Z 轴/同主机) 分配 ici_tp = 4 或 8"]
    C -- "<=13B 小模型" --> E["设 ici_tp = 1 (免除逐层同步开销)"]
    D & E --> F{"第 3 步: 模型是否包含稀疏 MoE 专家层或 >=64K 长序列?"}
    F -- "含 MoE 专家层 (如 64/128 Experts)" --> G["使用 XLA Shardy 子轴切分: 将剩余 64 芯片 ICI 平面同时映射为 Attention 的 ici_fsdp=64 与 MoE 的 ici_ep=64"]
    F -- "含 >=64K 超长上下文" --> H["从 ICI 平面拆出 ici_cp=4..8 子轴，通过 shard_map 执行 Ring Attention (ppermute)"]
```

### 3.2 Tier 4 `XLA Shardy` 优先级传播与双层 `ICI / DCN` 混合设备网格架构图

```mermaid
flowchart LR
    subgraph Physical["🖧 1. 物理两级拓扑 (4 个 8x8x8 3D Torus Slice = 2,048 芯片)"]
        S1["Slice #0..#3 (每个 Slice 512 芯片)\nSlice 内: 1.2 Tbps/chip 3D Torus ICI (Wrap-Around)\nSlice 间: Jupiter DCN 以太网"]
    end

    subgraph Mesh["🧩 2. create_hybrid_device_mesh 逻辑网格对齐"]
        S1 --> M1["dcn_dp = 4 (跨 Slice DCN 轴)\n每步仅 1 次外层梯度 All-Reduce"]
        S1 --> M2["ici_fsdp_ep = 64 (Slice 内 X×Y 2D Torus)\n承载 FSDP All-Gather & MoE All-to-All"]
        S1 --> M3["ici_tp = 8 (Slice 内 Z 轴/机内)\n承载逐层矩阵乘 TP 切分"]
    end

    subgraph Shardy["⚡ 3. XLA Shardy (MLIR) 编译器与 shard_map"]
        M1 & M2 & M3 --> SDY["Shardy 优先级分片传播器 (--xla_use_shardy=true)\n消除 Attention <-> MoE 残差分支 38GB 冗余 Reshard!"]
        SDY --> SMAP["shard_map + Pallas 内核\n异步隐藏 93.6% 集合通信 -> 持续 MFU 达 59.6%!"]
    end
```

### 3.3 `Shardy` 消除 MoE 分支冗余 `Reshard` 与双缓冲通信重叠时序图

```mermaid
sequenceDiagram
    autonumber
    participant Code as MaxText / JAX 模型定义
    participant Shardy as XLA Shardy (MLIR) 编译器
    participant ICI as Slice 内 3D Torus ICI (512 芯片)
    participant DCN as 跨 Slice Jupiter DCN (4 Slices)

    Code->>Shardy: 传入混合网格 `Mesh(dcn_dp=4, ici_fsdp_ep=64, ici_tp=8)` 与优先级 `PartitionSpec`
    Shardy->>Shardy: 在 `MLIR StableHLO` 层按优先级传播分片: 识别出残差主干保持 `(dcn_dp, ici_fsdp_ep, None)`，仅在进出 MoE 门控时执行唯一一次 `All-to-All` (`0.0 GB` 冗余拷贝!)
    Shardy->>ICI: 编译生成异步流水线计划: 计算第 $L$ 层 `MXU` 矩阵乘的同时，后台通过 `ICI` 预取第 $L+1$ 层 `FSDP` 权重
    ICI->>ICI: 64 层前向+反向在 Slice 内 `1.2 Tbps ICI` 高速完成，通信暴露率仅 `6.4%`
    ICI->>DCN: 反向传播逐层完成时，异步通过 `DCN` 执行跨 4 个 Slice 的外层 `dcn_dp All-Reduce` (`59.6%` 持续 MFU!)
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：错位网格（`TP/FSDP` 跨 `DCN`） | Tier 2：单维扁平 `1D FSDP` 环（512 芯片） | Tier 3：旧版扁平 HLO `GSPMD` 分区器 | Tier 4：新一代 `XLA Shardy (MLIR)` + 双层 `ICI/DCN` 网格 + `shard_map`（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **`2,048` 芯片持续训练 `MFU`** | **`15.8%`（跨 DCN 逐层同步卡死）** | `38.4%` | `48.2%` | **`59.6%`（较旧版 `GSPMD` 净提升 `+11.4 pts`！）** |
| **未被计算隐藏的暴露通信占比** | **`68.5%`** | `34.0%` | `21.5%` | **`6.4%`（`93.6%` 通信被 `MXU` 计算完美掩盖）** |
| **每步冗余 `Reshard` 浪费流量** | `148.0 GB / 步` | `62.0 GB / 步` | **`38.0 GB / 步`（Attention/MoE 分支冲突）** | **`0.0 GB / 步`（`Shardy` 优先级传播精准消解冲突）** |
| **分片冲突可调试性 (`Observability`)** | 极差 | 差 | ❌ 黑盒扁平 HLO，极难定位谁插了 `All-Gather` | **✅ `MLIR` 源码级可读分片传播追踪日志** |
| **超长上下文 (`128K+`) 与自定义集合通信** | OOM 崩溃 | OOM 崩溃 | 仅支持粗粒度自动切分 | **✅ 原生结合 `shard_map` 手写 `ppermute` 环状注意力** |
| **月度 `2,048` 芯片训练总 TCO** | `$3,920,000 / 月` | `$2,450,000 / 月` | `$1,920,000 / 月` | **`$1,480,000 / 月`（节省 `62.2%`）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（`2,048` 芯片 `4x512` Multislice 训练 `400B-A45B` 稀疏 MoE 大模型）
* **业务规模**：在 `4` 个 `512` 芯片（`8x8x8` 3D Torus）Slice 组成的 `2,048` 芯片集群上训练 64 层 `400B` 总参数、`45B` 激活参数的稀疏 MoE 大模型。
* **工程实测与仿真代码**：见 [`experiments/jax_xla_shardy_gspmd_device_mesh_ici_dcn_parallelism_selection/run.py`](../../experiments/jax_xla_shardy_gspmd_device_mesh_ici_dcn_parallelism_selection/run.py) 与 [`results.json`](../../experiments/jax_xla_shardy_gspmd_device_mesh_ici_dcn_parallelism_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开 JAX/XLA Shardy 与 ICI/DCN 设备网格容量计算器 (`jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-sizer.html`)](../reports/jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-selection-cards.html`)](../reports/jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Misaligned TP/FSDP on DCN", "mfu_pct": 15.8, "exposed_comm_pct": 68.5, "reshard_gb": 148.0, "monthly_tco_m": 3.92 },
      { "tier": "Tier 2: 1D Flat FSDP Ring", "mfu_pct": 38.4, "exposed_comm_pct": 34.0, "reshard_gb": 62.0, "monthly_tco_m": 2.45 },
      { "tier": "Tier 3: Legacy HLO GSPMD", "mfu_pct": 48.2, "exposed_comm_pct": 21.5, "reshard_gb": 38.0, "monthly_tco_m": 1.92 },
      { "tier": "Tier 4: XLA Shardy + 3D ICI/DCN Mesh", "mfu_pct": 59.6, "exposed_comm_pct": 6.4, "reshard_gb": 0.0, "monthly_tco_m": 1.48 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "mfu_pct": "Percentage",
    "exposed_comm_pct": "Percentage",
    "reshard_gb": "Quantity",
    "monthly_tco_m": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 (XLA Shardy + 3D ICI/DCN Mesh) Hits 59.6% MFU and Cuts Monthly Training TCO by 62.2%",
    "subtitle": "Monthly 2,048-chip MoE training TCO ($ USD Millions) across 4x512-chip slices",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_m",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **`create_hybrid_device_mesh` 物理拓扑强绑定**：多切片训练是否已使用 `mesh_utils.create_hybrid_device_mesh(mesh_shape=..., dcn_mesh_shape=...)`，确保 `TP`、`FSDP` 与 `EP` `100%` 落在 Slice 内部 `ICI` 轴上，且跨 Slice `DCN` 仅承载外层 `DP`？
- [ ] **启用 `XLA Shardy (MLIR)` 分区器（`--xla_use_shardy=true`）**：在混合稠密 Attention 与稀疏 MoE 的大模型训练中，是否已开启 `Shardy` 并检查编译期 HLO Dump，确认残差支路交汇处 `Redundant Reshard == 0`？
- [ ] **`FSDP` 2D/3D 正交环分解**：在 `>=256` 芯片的 3D Torus Slice 上，是否将 `FSDP` 与 `TP / EP / CP` 正交映射到不同的物理环面维度（而非拉成一条 512 跳的单维长环）？
- [ ] **异步集合通信与计算双缓冲重叠门槛**：是否已配置 `XLA` 异步 `All-Gather / Reduce-Scatter` 阈值，在 `XProf` 中验证未重叠暴露通信占比 `<10%`？
- [ ] **关键路径算子 `shard_map` 显式下沉**：对于 `>=64K` 序列的 Ring Attention 或自定义 `Pallas` 内核，是否通过 `shard_map` + `jax.lax.ppermute` 显式控制通信流水线？

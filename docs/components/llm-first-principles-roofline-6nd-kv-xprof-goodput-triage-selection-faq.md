# 大模型训练与推理第一性原理容量手算（`Roofline` 屋脊线 / `6ND` / `KV Cache`）、性能剖析（`XProf / Nsight` 四步排查树）与 `ML Productivity Goodput` 三乘积排障实战指南

> **适用读者**：需要为客户在白板上 5 分钟手算出千卡训练工期与推理并发卡数、并能看懂 `Google XProf` 或 `NVIDIA Nsight Systems` 火焰图定位 MFU 跌落根因的 Cloud & AI Infrastructure Solution Architect、性能工程负责人与大模型训推架构师。
> **核心课题**：当客户质疑 *“为什么我买了 `1,024` 颗最新芯片、监控面板显示硬件使用率（Duty Cycle）高达 `95%`，但训练 `70B` 模型 `15T` Tokens 的实际进度却比预期慢了 `4.5 倍`（真实有效算力产出仅 `12.5%`，每月白白烧掉 `$265 万`）？”* 时，架构师如何用 **第一性原理公式（`算术强度 AI 与 Roofline 拐点`、`6ND` 训练芯片天、`16 Bytes/Param` 显存账本、`KV Cache` 与 `TTFT/TPOT/MBU` 下界）** 当场算清理论天花板？如何通过 **`XProf / Nsight` 四步低 MFU 排查树（① 主机 Infeed 饥饿 $\rightarrow$ ② XLA 重编译/碎核 $\rightarrow$ ③ 未重叠集合通信暴露 $\rightarrow$ ④ `128x128` 脉动阵列填充浪费）** 将持续 MFU 从 `21.5%` 提升至 **`59.2%`**，并结合 **Google `ML Productivity Goodput` 三乘积拆解（`调度效率 × 运行期可用率 × 有效步进率 [剔除回滚 Badput]`）** 将 Goodput 从 `58.0%` 推升至 **`96.5%`**（真实集群总产出提升 **`4.57 倍`：`12.5% -> 57.1%`**，月度无效算力浪费降低 **`78.1%`**）？

![大模型训练与推理第一性原理容量手算、XProf 性能剖析与 Goodput 三乘积排障总结图](../../assets/svg/llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在千卡智算集群里，**最具有欺骗性的假指标就是“芯片硬件忙闲率（`GPU Utilization / TPU Duty Cycle = 95%`）”——一颗正在死等跨机 `AllReduce` 同步屏障、或者正在重算故障崩溃前 30 分钟白跑步骤（`Badput`）的芯片，硬件监控同样显示 95% 繁忙，但对业务的实际贡献是 0**！真正的集群产出必须用 **`真实集群算力转化率 (Real Cluster Yield) = MFU × ML Productivity Goodput`** 衡量：

| 容量规划与性能调优层级 | 适用阶段 | 持续训练算力利用率 (`MFU`) | 端到端有效进度率 (`3-Factor Goodput`) | 真实集群算力转化率 (`MFU × Goodput`) | 在线推理显存带宽利用率 (`Decode MBU`) | 月度无效空转与浪费成本 (`Waste TCO`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：盲目拍脑袋堆卡（无 Roofline 测算、动态 Shape、P/D 混部，反模式）** | ❌ **严禁用于百卡以上生产集群** | `21.5%` | `58.0%`（大量回滚重算） | **`12.5%`（`87.5%` 算力打水漂！）** | `28.0%` | `$2,650,000 / 月` |
| **Tier 2：仅看粗粒度芯片使用率监控（忽视回滚 `Badput` 与暴露通信）** | 缺乏 `XProf` 剖析能力的基础运维团队 | `36.0%` | `74.5%` | `26.8%` | `45.0%` | `$1,720,000 / 月` |
| **Tier 3：仅做静态 Shape 分桶（未做异步通信重叠与 Prefill/Decode 分离）** | 单机或小规模集群基线 | `48.5%` | `84.0%` | `40.7%` | `58.0%` | `$1,180,000 / 月` |
| **Tier 4：第一性原理 `Roofline/6ND` 定容 + `XProf` 异步通信重叠 + 三乘积 `Goodput` + `P/D` 分离（推荐）** | **千卡/万卡 TPU & GPU 旗舰训推主力标准** | **`59.2%`** | **`96.5%`** | **`57.1%`（较 Tier 1 提升 `4.57 倍`！）** | **`78.5%`** | **`$580,000 / 月`（省 `78.1%`）** |

### 💡 架构师随身必背的五大第一性原理公式（白板 5 分钟手算卡）

#### 公式 1：算术强度（Arithmetic Intensity）与 `Roofline` 屋脊拐点（Ridge Point）
$$\text{Arithmetic Intensity (AI)} = \frac{\text{FLOPs Executed}}{\text{Bytes Moved Across HBM}} \quad (\text{FLOPs/Byte}), \qquad \text{AI}_{\text{ridge}} = \frac{\text{Peak Compute (FLOPS)}}{\text{Peak HBM Bandwidth (Bytes/s)}}$$
* **主流芯片屋脊拐点速查**：
  * **`TPU7x Ironwood`**：`FP8` 拐点 = $4,614\text{ TF} / 7.37\text{ TB/s} \approx \mathbf{626\text{ FLOPs/Byte}}$；`BF16` 拐点 $\approx \mathbf{313\text{ FLOPs/Byte}}$。
  * **`TPU v6e (Trillium)`**：`BF16` 拐点 = $918\text{ TF} / 1.64\text{ TB/s} \approx \mathbf{560\text{ FLOPs/Byte}}$。
  * **`NVIDIA H200`**：`BF16` 拐点 = $989\text{ TF} / 4.8\text{ TB/s} \approx \mathbf{206\text{ FLOPs/Byte}}$；`FP8` 拐点 $\approx \mathbf{412\text{ FLOPs/Byte}}$。
  * **`NVIDIA B200`**：`BF16` 拐点 = $2,250\text{ TF} / 8.0\text{ TB/s} \approx \mathbf{281\text{ FLOPs/Byte}}$；`FP8` 拐点 $\approx \mathbf{562\text{ FLOPs/Byte}}$。
* **物理推论**：矩阵乘 $C_{B \times N} = A_{B \times K} \times W_{K \times N}$ 在大权重主导时的算术强度约为 $\text{AI} \approx \frac{2 B K N}{2 K N} = B \text{ (BF16)}$（`FP8` 下约为 $2B$）。因此，**当有效并发 Token 数 $B < \text{AI}_{\text{ridge}}$（例如自回归 Decode 阶段每步每请求仅生成 1 个 Token，$B \ll 300$）时，系统 100% 处于显存带宽受限区（Memory-Bandwidth-Bound）；只有 Prefill 阶段或大 Batch 训练（$B \times S \ge 2,048$）才能越过屋脊拐点进入算力受限区（Compute-Bound）**！

#### 公式 2：`6ND` 大模型预训练芯片天与工期手算公式
$$\text{Total FLOPs} \approx 6 \times N_{\text{active\_params}} \times D_{\text{tokens}}, \qquad \text{Training Days} = \frac{6 \times N \times D}{C_{\text{chips}} \times \text{Peak FLOPS}_{\text{chip}} \times \mathbf{MFU} \times \mathbf{Goodput} \times 86,400}$$
* **为什么是 `6ND`？**：前向传播每个参数对每个 Token 做 1 次乘加（`2ND` FLOPs），反向传播需要算对激活值梯度（`2ND`）与对权重梯度（`2ND`），合计 **`6ND`**（若开启重计算 Activation Checkpointing / Rematerialization，则约为 **`8ND`**）。
* **白板算例**：用 `1,024` 颗 `BF16` 峰值 `2,307 TFLOPS` 的芯片训练 `70B` 稠密模型 `15T` Tokens（总算量 $6 \times 70\times 10^9 \times 15\times 10^{12} = 6.3\times 10^{24}\text{ FLOPs}$）：
  * Tier 1（`MFU 21.5% × Goodput 58.0% = 12.5%` 产出）：耗时 **`69.5 天`**；
  * Tier 4（`MFU 59.2% × Goodput 96.5% = 57.1%` 产出）：耗时仅 **`15.2 天`**（工期缩短 **`54.3 天`**！）。

#### 公式 3：混合精度训练显存账本（`16 Bytes/Param` 法则）
$$\text{Model \& Optimizer HBM} = N_{\text{params}} \times (\underbrace{2\text{B}}_{\text{BF16 权重}} + \underbrace{2\text{B}}_{\text{BF16 梯度}} + \underbrace{4\text{B} + 4\text{B} + 4\text{B}}_{\text{FP32 主权重 + Adam } m, v}) = \mathbf{16 \times N_{\text{params}}\text{ Bytes}}$$
* `70B` 模型静态训练状态需 $70\times 16 = \mathbf{1,120\text{ GB}}$ HBM，必须通过 `FSDP / ZeRO-3` 加上 `TP` 切分到至少 `16–32` 颗芯片上，留出余量存放激活值（Activations）。

#### 公式 4：在线推理 `KV Cache` 显存与 `TTFT / TPOT / MBU` 物理下界
$$\text{KV Bytes per Token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$
$$\text{TTFT}_{\text{prefill}} \approx \frac{2 \times N_{\text{active}} \times S_{\text{prompt}}}{C_{\text{tp}} \times \text{Peak FLOPS} \times \text{MFU}}, \qquad \text{TPOT}_{\text{decode}} \ge \frac{\text{Weight Bytes} + B \times S_{\text{ctx}} \times \text{KV Bytes/Token}}{C_{\text{tp}} \times \text{HBM\_BW} \times \mathbf{MBU}}$$
* **白板算例（`Llama-3-70B`，80 层，$n_{\text{kv}}=8, d_{\text{head}}=128$，`BF16`）**：每个 Token 的 KV Cache = $2 \times 80 \times 8 \times 128 \times 2 = \mathbf{327.68\text{ KB/Token}}$（`32K` 上下文单并发占 **`10.48 GB`**；改用 `FP8 KV` 直接减半至 **`5.24 GB`**）。

#### 公式 5：Google `ML Productivity Goodput` 三乘积分解公式
$$\text{ML Goodput} = \underbrace{\text{Scheduling \& Provisioning Eff.}}_{\text{排队/拉镜像/凑卡成组占比}} \times \underbrace{\text{Runtime Uptime Eff.}}_{\text{剔除宕机重启与同步存盘停顿}} \times \underbrace{\text{Step Progress Eff.}}_{1 - \frac{\text{故障回滚丢失重算步数 (Badput)}}{\text{总运行步数}}}$$

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：当客户训练任务的 `MFU` 只有 `21.5%` 时，架构师打开 `Google XProf (TensorBoard Profile)` 或 `NVIDIA Nsight Systems` 应该按什么顺序在 10 分钟内抓出真凶？
* **四步确定性低 MFU 排查树（4-Step Low-MFU Triage Tree）**：
  1. **看第 1 处：`Host-to-Device Infeed` 是否有空隙（数据加载饥饿）？**
     * 在 `XProf Trace Viewer` 中查看每步开始前是否有毫秒级的空白等待。如果有，说明 CPU 数据加载器（`Grain / PyTorch DataLoader`）或未加本地 NVMe 缓存的 `GCS FUSE` 喂不饱芯片！**解法**：开启 `GCS FUSE` 本地 NVMe 缓存 + 多进程预取（`prefetch_to_device`）。
  2. **看第 2 处：是否存在周期性 `30–60 秒` 断崖停顿（`XLA Recompilation`）？**
     * 检查日志与 Trace 是否出现 `Compiling HLO module...`。如果有，说明存在动态序列长度或 Python 标量分支。**解法**：强制做幂次长度分桶（Bucketing）与序列拼接（Sequence Packing）。
  3. **看第 3 处：集合通信条（`All-Gather / Reduce-Scatter / All-to-All`）是否暴露在 `MXU / TensorCore` 计算条之外？**
     * 在理想状态下，通信通道（`ICI / NVLink`）的色块应该与计算通道（`MXU / SM`）上下完全重叠（Overlapped）。如果通信条单独暴露占了步时的 `40%`，要么是因为你把 `TP / FSDP` 错误地放到了跨切片慢速 `DCN` 上，要么是因为申请了非 `4x4x4` 回环切片，或者未开启异步通信流水线（`async_collective_permute`）！
  4. **看第 4 处：单个矩阵乘 Kernel 的 `MXU / TensorCore` 饱和度是否过低（Tile Padding 浪费）？**
     * TPU `MXU` 的物理硬件基本块是 `128x128`（或 `256x256`）。如果词表大小（Vocab Size）、隐藏层维度或单卡 Micro-batch 尺寸不是 **`128 / 256` 的整数倍**，编译器会自动补零（Padding），导致高达 `30%` 的脉动阵列单元在算无效的 `0 × 0`！

### Q2：为什么说在线推理的 Decode 阶段看 `MFU` 是外行，必须看 `MBU（Memory Bandwidth Utilization，显存带宽利用率）` 与 `P/D 分离`？
* **Prefill 拼的是 TFLOPS（算力），Decode 拼的是 TB/s（显存带宽）**：
  * 在 Decode 阶段，哪怕只有 16 个并发用户，每生成一个新 Token 都必须把整个 `70B` 模型的权重（`FP8` 下为 `70 GB`）连同 16 个用户的 `KV Cache` 从 HBM 完整读进计算核一次！此时算术强度仅 `~16 FLOPs/Byte`（远低于 `TPU7x` 的 `626` 或 `H200` 的 `206` 拐点），`MFU` 只有可怜的 `3%–5%`，但这完全正常——此时必须考核 **`MBU`（实测达成的有效显存带宽 / 硬件标称 HBM 带宽，优秀目标 `>=75%`）**。
  * 如果把长 Prompt 的 Prefill 请求与正在生成的 Decode 请求混在同组卡上（Co-located），一个 `32K` Prefill 插入进来就会独占计算核 `400ms`，把同批次所有用户的 Decode `TPOT` 尾延迟直接拉爆！**将 Prefill 与 Decode 物理分离（P/D Disaggregation）并通过 `ICI / RDMA` 传输 KV Cache，可将 Decode `MBU` 推升至 `78.5%`、达标吞吐翻倍**。

---

## 3. 🧭 直观可视化图解（Mermaid 排查树 + Roofline 架构 + 时序图）

### 3.1 `XProf / Nsight` 四步低 MFU 诊断与优化决策树

```mermaid
flowchart TD
    A["告警: 集群持续 MFU < 35% 或真实产出 (MFU x Goodput) 偏低"] --> B{"第 1 步: 检查 XProf Input Pipeline / Host Infeed Gap?"}
    B -- "Infeed 空置占比 > 5%" --> B1["修复存储与加载: 开启 GCS FUSE 本地 NVMe 缓存 + Grain 多进程预取"]
    B -- "Infeed 正常 (<1%)" --> C{"第 2 步: 检查 XLA Compilation Cache Miss / 动态 Shape?"}
    C -- "运行期发生 HLO 重编译" --> C1["修复动态图: 对 (Batch, Seq_Len) 幂次分桶 + 消除步内 Python .item() 同步"]
    C -- "0 重编译" --> D{"第 3 步: 检查 Collective (AllGather/ReduceScatter/AllToAll) 暴露比例?"}
    D -- "通信未与 MXU/TensorCore 重叠 (>15% 暴露)" --> D1["修复拓扑与并行轴:\n① 确保 4x4x4 整数倍激活 Torus 回环\n② TP/FSDP 锁在 ICI，仅外层 DP 走 DCN\n③ 开启 Shardy/NCCL 异步双缓冲通信重叠"]
    D -- "通信已隐藏" --> E["第 4 步: 检查矩阵维度与 Micro-Batch 算术强度\n将 Vocab/Hidden/Batch 对齐至 128/256 整数倍以消除 MXU Tile 补零浪费 -> MFU 达 59.2%!"]
```

### 3.2 第一性原理 `Roofline` 边界与 `Google ML Productivity Goodput` 三乘积全栈架构图

```mermaid
flowchart LR
    subgraph Roofline["📐 1. 单步硬件效率层 (Single-Step MFU & MBU)"]
        PRE["训练与 Prefill (Compute-Bound)\n算术强度 AI >= Ridge Point (206~626)\n目标: 压榨 MXU/TensorCore MFU -> 59.2%"]
        DEC["自回归 Decode (Bandwidth-Bound)\n算术强度 AI << Ridge Point\n目标: P/D 分离压榨 HBM MBU -> 78.5%"]
    end

    subgraph Goodput["⏱️ 2. Google ML Productivity Goodput 三乘积层 (96.5%)"]
        G1["因子 1: 调度与供给效率 (99.1%)\nKueue TAS 成组交付 + <35s 镜像流"] --> G2["因子 2: 运行期在线率 (98.8%)\nOCS 秒级绕障 + 2.4s Orbax 异步存盘"]
        G2 --> G3["因子 3: 有效步进率 (98.6%)\n本地 NVMe <5s 紧急快照消灭回滚 Badput"]
    end

    Roofline & Goodput ==> YIELD["🏆 真实集群有效算力产出\nReal Yield = MFU (59.2%) x Goodput (96.5%) = 57.1% (4.57x 提升!)"]
```

### 3.3 `XProf` 剖析闭环与消除未重叠通信及回滚 `Badput` 时序图

```mermaid
sequenceDiagram
    autonumber
    participant Eng as 架构师 / 性能工程师
    participant XProf as Google XProf / Cloud Monitoring Goodput SDK
    participant Job as 1,024 芯片训练作业 (MaxText / JAX)
    participant Ckpt as Orbax Async + Local NVMe

    Job->>XProf: 上报初始基线: 硬件繁忙度 `95%`，但实测 `MFU = 21.5%`, `ML Goodput = 58.0%` (`Real Yield = 12.5%`)
    Eng->>XProf: 打开 `XProf Trace Viewer` & `Goodput Breakdown Dashboard`
    XProf-->>Eng: 诊断发现: ① `All-Gather` 跨非回环边界暴露占步时 `38%`；② Vocab `32001` 未对齐 `256`；③ 30m 同步存盘 + 故障回滚丢掉 `42%` Goodput
    Eng->>Job: 整改 1: 切换 `4x4x8` 完整回环切片 + 开启 `Shardy` 异步 `All-Gather` 与 `MXU` 双缓冲重叠 + Vocab 补齐至 `32256`
    Eng->>Ckpt: 整改 2: 启用 `Orbax` 2.4s 异步存盘 + 本地 NVMe `<5s` 紧急快照
    Job->>XProf: 重新采集剖析: 单步 `MFU` 跃升至 `59.2%`，三乘积 `ML Goodput` 跃升至 `96.5%`
    XProf-->>Eng: 确认真实集群产出达 `57.1%` (`70B/15T` 总工期从 `69.5 天` 缩短至 `15.2 天`)
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：盲目堆卡（无 Roofline / 未分桶 / 混部） | Tier 2：仅看粗粒度芯片使用率监控 (`Duty Cycle`) | Tier 3：仅做静态 Shape 分桶（无异步重叠与 P/D 分离） | Tier 4：`Roofline/6ND` 定容 + `XProf` 异步重叠 + 三乘积 `Goodput` + `P/D` 分离（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **持续训练算力利用率 (`MFU`)** | `21.5%`（通信暴露 + Tile 补零浪费） | `36.0%` | `48.5%` | **`59.2%`（`XProf` 四步调优 + 异步集合通信完全隐藏）** |
| **端到端有效进度率 (`3-Factor Goodput`)** | `58.0%`（同步存盘冻结 + 崩溃回滚重算） | `74.5%` | `84.0%` | **`96.5%`（`2.4s` 异步存盘 + `<5s` 紧急快照零回滚）** |
| **真实集群算力转化率 (`MFU × Goodput`)** | **`12.5%`（87.5% 算力预算被白白蒸发！）** | `26.8%` | `40.7%` | **`57.1%`（较 Tier 1 提升 `4.57 倍`！）** |
| **`70B` 模型 `15T` Tokens (`1,024` 卡) 工期** | **`69.5 天`** | `32.4 天` | `21.3 天` | **`15.2 天`（节省 `54.3 天` 研发上市时间！）** |
| **在线推理 Decode 显存带宽利用率 (`MBU`)** | `28.0%`（被长 Prefill 抢占阻塞） | `45.0%` | `58.0%` | **`78.5%`（P/D 物理分离 + `FP8 KV Cache`）** |
| **月度无效空转与浪费总成本 (`Waste TCO`)** | `$2,650,000 / 月` | `$1,720,000 / 月` | `$1,180,000 / 月` | **`$580,000 / 月`（减少 `78.1%` 算力浪费）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（`1,024` 颗芯片训练 `70B` 模型 `15T` Tokens + `32K` 上下文在线推理）
* **业务规模**：企业使用 `1,024` 颗旗舰加速器（如 `TPU7x Ironwood` 或 `H200`）执行 $N = 70\text{B}$ 参数、$D = 15\text{T}$ Tokens 的预训练（总浮点计算量 $6ND = 6.3\times 10^{24}\text{ FLOPs}$），并承载 `32K` 上下文在线推理服务。
* **工程实测与仿真代码**：见 [`experiments/llm_first_principles_roofline_6nd_kv_xprof_goodput_triage_selection/run.py`](../../experiments/llm_first_principles_roofline_6nd_kv_xprof_goodput_triage_selection/run.py) 与 [`results.json`](../../experiments/llm_first_principles_roofline_6nd_kv_xprof_goodput_triage_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开大模型第一性原理容量（Roofline / 6ND / KV Cache）与 Goodput 计算器 (`llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-sizer.html`)](../reports/llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-selection-cards.html`)](../reports/llm-first-principles-roofline-6nd-kv-xprof-goodput-triage-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Blind Scaling (12.5% Yield)", "mfu_pct": 21.5, "goodput_pct": 58.0, "real_yield_pct": 12.5, "waste_tco_m": 2.65 },
      { "tier": "Tier 2: Chip-Busy Monitoring (26.8% Yield)", "mfu_pct": 36.0, "goodput_pct": 74.5, "real_yield_pct": 26.8, "waste_tco_m": 1.72 },
      { "tier": "Tier 3: Static Bucketing Only (40.7% Yield)", "mfu_pct": 48.5, "goodput_pct": 84.0, "real_yield_pct": 40.7, "waste_tco_m": 1.18 },
      { "tier": "Tier 4: Roofline + XProf + Goodput (57.1% Yield)", "mfu_pct": 59.2, "goodput_pct": 96.5, "real_yield_pct": 57.1, "waste_tco_m": 0.58 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "mfu_pct": "Percentage",
    "goodput_pct": "Percentage",
    "real_yield_pct": "Percentage",
    "waste_tco_m": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Lifts Real Cluster Yield 4.57x (12.5% to 57.1%) and Cuts Monthly Compute Waste by 78.1%",
    "subtitle": "Monthly wasted cluster compute and serving TCO ($ USD Millions) on a 1,024-chip fleet",
    "encodings": {
      "x": "tier",
      "y": "waste_tco_m",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **立项前第一性原理三表手算（`6ND` 工期表 + `16B/Param` 显存表 + `Roofline` 拐点表）**：任何百卡以上训推立项是否已先用公式算出理论最小芯片数、HBM 激活余量、期望 `MFU/MBU` 及理论 `TTFT / TPOT` 下界？
- [ ] **废弃单纯“硬件繁忙率”指标，强制上线 `MFU × 3-Factor ML Goodput` 看板**：是否在 Cloud Monitoring / Prometheus 中接入 `ml-goodput-measurement` SDK，实时拆解监控 `调度等待损耗时`、`存盘/重启停顿耗时` 与 `故障回滚重算耗时 (Badput)`？
- [ ] **`XProf / Nsight` 四步剖析门禁**：每个新模型上线前是否已采集至少 5 步 `XProf / Nsight` Trace，确认 `Infeed Wait < 1%`、`XLA Recompiles == 0`、且 `All-Gather / Reduce-Scatter` 与计算重叠率 `>=85%`？
- [ ] **张量维度 `128 / 256` 对齐检查**：模型的 `vocab_size`、`hidden_size`、`intermediate_size` 与每芯片 `micro_batch_size × seq_len` 是否全部对齐至 `128` 或 `256` 的整数倍，避免脉动阵列（`MXU`）Tile 补零浪费？
- [ ] **在线推理 Prefill/Decode 屋脊解耦**：对于长上下文（`>=16K`）工作负载，是否已将算力受限的 Prefill 与显存带宽受限的 Decode 物理分离（或启用 Chunked Prefill + `FP8 KV Cache`），确保 Decode `MBU >= 75%`？

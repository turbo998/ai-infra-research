# 网内计算 SHARP 集合通信、NVLS 多播硬件归约与浮点非结合律位级漂移边界 (`multi-tier-collective-sharp-ar`)

> **证据类型声明**：本文基于 `NVIDIA/nccl` 主干源码（commit `3e46c089f4ae4732722a419eb6096aead7e47a30`, `published_at: 2026-10-08`）的 `src/transport/nvls.cc`、`src/transport/coll_net.cc`、`src/tuning/nvls.cc`、`src/tuning/collnet.cc`、`src/tuning/tuning.cc` 公开实现核验（`implementation`），并结合本项目原创 Python 标准库离线可复现实验 `experiments/sharp_in_network_collectives/run.py`（`own-offline`）构建跨规模集合通信带宽、交换机硬件树饱和回退与浮点位级确定性模型。本文不包含未授权真实万卡 IB/UEC 交换机硬件实测数据。

---

## 1. 为什么万卡同步训练的 AllReduce 必须从端到端 Ring 走向多级网内计算（In-Network Computing）？

在 `64–16,384` 卡的大规模 FSDP2 / 数据并行或 Context Parallelism 同步归约中，集合通信面临两重物理天花板：

1. **流量放大系数（Bus Bandwidth Factor）瓶颈**：
   - 在传统 **Ring AllReduce**（`NCCL_ALGO_RING`）或 **Double Binary Tree**（`NCCL_ALGO_TREE`）中，为了完成 `ReduceScatter + AllGather` 等价的全归约，每个节点必须向网络发送并接收接近 $2 \times \frac{N-1}{N} \approx 2.0\times$ 的梯度负载字节数。
   - 当集群具备 **机内 NVLink SHARP（`NVLS`）+ 机间 InfiniBand/UEC 交换机树归约（`CollNet` / `IB SHARP v3/v4`）** 时：
     - **机内阶段**：GPU 利用 NVSwitch 硬件多播与网内 ALU（`multimem.ld_reduce` / `multimem.st`）以单次写/读完成 8 卡或 72 卡机内归约；
     - **机间阶段**：每个节点仅向 Top-of-Rack (ToR) / Spine 交换机树上传 **$1.0\times$** 归约数据，交换机 ASIC 在数据面直接对多端口到达的数据块执行硬件浮点累加，随后以 **$1.0\times$** 硬件多播下发全局结果。机间网络注入字节数直接腰斩（**理论算法带宽上限提升 $2\times$**）。

2. **跳数延迟（Hop Latency）与流水线气泡瓶颈**：
   - Ring AllReduce 需要 $2(N-1)$ 个串行跳步，在 `16,384` 卡（`2,048` 台 8 卡节点）规模下高达 `32,766` 步，微小的单跳 RTT 抖动或 PFC 停顿会沿环路级联放大。
   - 树形归约与交换机 SHARP 将跨节点跳步压缩至 $2 \log_2(N_{\text{nodes}})$（`2,048` 节点仅需 `22` 个树跳步）。

---

## 2. 一手源码走读：`NVIDIA/nccl` 中的 `NVLS`、`NVLS_TREE` 与 `CollNet` 代价模型

### 2.1 `src/tuning/nvls.cc`（blob `61e1e938eb79a5659dd08b2b5d859bf83a149b72`）：架构效率因子与跨节点带宽公式

在 `src/tuning/nvls.cc` 中，NCCL 显式定义了各代 GPU 架构的 NVLS 硬件多播效率折损因子 `nvlsEfficiency`：

```cpp
// src/tuning/nvls.cc (lines 11-16)
static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
  0.0f,  // Volta (unsupported)
  0.0f,  // Ampere (unsupported)
  0.85f, // Hopper (SM90, NVSwitch Gen3)
  0.74f, // Blackwell (SM100, NVSwitch Gen4 / NVL72)
};
```

在 `ncclTuningNvlsModelInit()` 中，NCCL 严格区分了两种基于 NVLS 的跨节点算法：
- **`NCCL_ALGO_NVLS`（机内 NVLS + 机间 CollNet IB SHARP）**：
  - 当 `comm->nNodes > 1` 且 `comm->config.collnetEnable == 0` 时，`NCCL_ALGO_NVLS` 被硬性禁用（`lines 36-39`）；
  - 延迟模型仅需一次机内延迟加一次机间 SHARP 树往返：`generalLatencies = intraLat + interLat`（`lines 98-100`）；
- **`NCCL_ALGO_NVLS_TREE`（机内 NVLS + 机间主干双二叉树，无需交换机 SHARP）**：
  - 仅在 `comm->nNodes > 1` 且函数为 `ncclFuncAllReduce` 时启用（`lines 31-34, 52-55`）；
  - 当节点数 `comm->nNodes <= 2` 时，双树互补可获得 $2\times$ 机间带宽（`interBw = bwInter * 2`），而当 `comm->nNodes > 2` 时受限于树父子双向流量，`interBw = bwInter * 1`（`line 65`）；
  - 延迟随节点数对数增长：`generalLatencies += intraLat + 2 * log2i(comm->nNodes) * interLat`（`lines 101-103`）。

此外，在 `src/transport/nvls.cc`（blob `155f6adf30223e19fb52fa4ed1c30b2014f27c44`）中：
- NVLS 物理显存缓冲区强制要求 **`NVLS_MEM_ALIGN_SIZE = (1 << 21)`（即 `2 MiB` 物理页对齐）**，通过 `cuMemCreate` + `cuMulticastCreate` + `cuMulticastBindMem` 绑定到多播组；
- 针对 Blackwell `SM100` 多节点场景，`ncclNvlsTreeSm100Tuning` 按网络带宽档位（`>=50 GB/s` 400GbE/NDR 与 `<50 GB/s`）与节点规模（`2 / 4 / 8+` 节点）显式硬编码了 `chunkSize`（`64 KiB / 128 KiB / 256 KiB`）与 `nChannels`（`24 / 32` 通道），防止小包多播下 NVSwitch 信用（Credit）耗尽。

### 2.2 `src/tuning/collnet.cc`（blob `74bec57afc7afb54954456e7aa3f13e153cb18c9`）与 `src/transport/coll_net.cc`：Head Rank 约束与 Rail 拓扑门槛

在 `src/tuning/collnet.cc` 中，NCCL 对启用 `COLLNET_DIRECT` 或 `COLLNET_CHAIN` 施加了极其严格的硬件拓扑前置校验：
1. **Local Ranks $\le$ CollNet Heads 硬约束**（`lines 73-83`）：
   ```cpp
   for (int r = 0; r < comm->nRanks; r++) {
     int node = comm->rankToNode[r];
     if (comm->nodeRanks[node].localRanks > nHeads) {
       comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
       enabled[c] = 0; // Hard disable
       continue;
     }
   }
   ```
   这意味着在 8 卡节点上，如果某个任务只分配了少于每节点本地 GPU 数的 NIC Rail（或部分 NIC 未接入 SHARP 域，导致 `localRanks > collNetHeadsNum`），`AllGather` / `ReduceScatter` 的 CollNet 支持会被**直接硬禁用**。
2. **串行化扇入延迟（Arity Serialization Latency）**：
   - `COLLNET_DIRECT` 在单节点多卡汇聚到 Head Rank 时，每增加一个本地 GPU 会叠加 `0.4 us` 的扇入串行化延迟（`line 105`: `(nRanks / nNodes - 1) * 0.4`），且本地 GPU 数超过 `NCCL_MAX_DIRECT_ARITY + 1`（即 `8` 卡）时直接失效（`line 130`）。
3. **有限的交换机硬件 SHARP Group 槽位**：
   - 在 `src/transport/coll_net.cc`（blob `91d65c58224902bb3c11ba7ce163d15a1f447654`）中，`COLLNET_GROUP_NSUBS = 8`，且交换机 ASIC 的并发活跃 SHARP Trees 数量受硬件表项限制（典型 InfiniBand Quantum-2/Quantum-X800 交换机单域支持约 `64` 个并发活跃 SHARP Group）。当多租户或多 DP/TP/CP 通信域并发申请超出上限时，后续 communicator 初始化 `collNetSupport` 失败，静默降级至 `NVLS_TREE` 或 `RING`。

---

## 3. 浮点非结合律（Floating-Point Non-Associativity）与网内归约位级漂移

在 IEEE-754 浮点数中，由于有限尾数位宽与舍入误差，加法**不满足结合律**：
$$(a \oplus b) \oplus c \neq a \oplus (b \oplus c)$$

- **为什么 Ring AllReduce 天然具有位级确定性（Bit-wise Reproducibility）？**
  在给定全局 Rank 拓扑时，Ring AllReduce 中每个块 $k$ 的累加顺序严格固定为 $\text{Rank}_{k} \to \text{Rank}_{k+1} \to \dots \to \text{Rank}_{k+N-1}$。只要输入张量相同，无论运行多少次，每位结果 100% 相同。
- **为什么交换机 SHARP 树归约会引发非确定性位级漂移？**
  1. 在多级交换机 SHARP 树中，ToR 与 Spine 交换机端口可能根据数据包实际到达微秒级先后顺序（Dynamic Arrival Order）或可变扇入树结构（Variable Arity Tree）对多路分支做归约；
  2. 如果交换机内部 ALU 采用与传输精度相同的窄位宽（如 `BF16` 逐级舍入累加），不同到达顺序会导致严重的尾数截断差异；
  3. 即便交换机 ASIC 内部使用 **FP32 累加器**（SHARP v3/v4 支持将 `BF16`/`FP16` 输入在交换机内晋升为 `FP32` 累加后再下发转回 `BF16`），如果父子树的结合顺序随网络拥塞动态重排，在病态正负抵消（Catastrophic Cancellation）梯度下仍可能出现 `1 ULP` 翻转。
  4. **确定性训练规避契约**：当开启 `torch.use_deterministic_algorithms(True)` 或需要严格 Bit-wise 复现（如 RL Rollout/Trainer 对齐、安全审计或 SDC 静默错误校验）时，必须固定 SHARP 树扇入槽位顺序（Fixed-Arity Static Slot Tree）并启用交换机 `FP32` 内部累加，或在校验步显式设置 `NCCL_COLLNET_ENABLE=0` 锁定 `NVLS_TREE` / `RING` 静态归约拓扑。

---

## 4. 原创离线实验验证（`experiments/sharp_in_network_collectives/`）

我们构建了确定性零依赖实验 `experiments/sharp_in_network_collectives/run.py`（种子 `42`，负载 `1 GiB` AllReduce，单节点 `8x GPU`，机内 NVSwitch `720 GB/s/GPU`，机间 `8x 400GbE/NDR = 360 GB/s/Node`），封印结果如下（`results.json`）：

### 4.1 `64–16,384` 卡下四类集合通信算法延迟与有效算法带宽 (`algbw`)

| 集群规模 (`n_gpus` / `n_nodes`) | `Ring` 延迟 / `algbw` | `Tree` 延迟 / `algbw` | `NVLS_TREE` 延迟 / `algbw` | `NVLS + SHARP` 延迟 / `algbw` | `SHARP` 相对 `Ring` 加速比 | `SHARP` 相对 `Tree` 加速比 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **64 GPUs** (`8` nodes) | `6.897 ms` / `155.69 GB/s` | `5.994 ms` / `179.14 GB/s` | `6.582 ms` / `163.14 GB/s` | **`4.066 ms` / `264.10 GB/s`** | **`1.70x`** | **`1.47x`** |
| **256 GPUs** (`32` nodes) | `8.634 ms` / `124.36 GB/s` | `6.648 ms` / `161.50 GB/s` | `7.209 ms` / `148.95 GB/s` | **`4.074 ms` / `263.54 GB/s`** | **`2.12x`** | **`1.63x`** |
| **1,024 GPUs** (`128` nodes) | `14.049 ms` / `76.43 GB/s` | `6.827 ms` / `157.29 GB/s` | `7.380 ms` / `145.49 GB/s` | **`4.083 ms` / `262.99 GB/s`** | **`3.44x`** | **`1.67x`** |
| **4,096 GPUs** (`512` nodes) | `34.378 ms` / `31.23 GB/s` | `6.885 ms` / `155.94 GB/s` | `7.437 ms` / `144.37 GB/s` | **`4.092 ms` / `262.43 GB/s`** | **`8.40x`** | **`1.68x`** |
| **16,384 GPUs** (`2,048` nodes) | `114.423 ms` / `9.38 GB/s` | `6.915 ms` / `155.29 GB/s` | `7.466 ms` / `143.82 GB/s` | **`4.100 ms` / `261.88 GB/s`** | **`27.91x`** | **`1.69x`** |

### 4.2 多租户交换机 SHARP 树耗尽（`max_groups = 64`）与回退尾延迟爆炸

在 `1,024` 卡规模下，当集群并发通信组/作业数跨越交换机硬件 `64` 个 SHARP Group 上限时：

| 并发作业数 (`concurrent_jobs`) | 获得硬件 SHARP 作业数 | 溢出回退作业数 (`fallback_jobs`) | 无管控回退至 `Ring` 的 P99 延迟 | 受控回退至 `NVLS_TREE` + QoS 隔离的 P99 延迟 | 无管控回退 P99 膨胀倍数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16 / 48 / 64** | `16 / 48 / 64` | `0` | `4.287 ms` | `4.287 ms` | `1.05x` |
| **80 jobs** | `64` | `16` | **`15.875 ms`** | **`7.557 ms`** | **`3.89x`** |
| **128 jobs** | `64` | `64` | **`18.615 ms`** | **`7.823 ms`** | **`4.56x`** |

### 4.3 `64` Ranks 浮点非结合律位级漂移实测（`512` 组病态梯度随机树排列试验）

- **交换机逐级 `BF16` 累加（动态到达顺序）**：在 `512` 组试验中，**`100.0%` 的试验出现位级不一致**，平均最大漂移达 **`357.79 ULP`**，最坏单次漂移高达 **`48,704 ULP`**（正负大数抵消顺序改变导致指数阶跃）。
- **交换机内部 `FP32` 累加后转回 `BF16`（`SHARP v3/v4 FP32 pipeline`）**：位级不一致率从 `100.0%` 骤降至 **`0.0%`**。
- **固定槽位顺序静态树归约（Fixed-Arity Static Slot Tree）**：位级不一致率严格为 **`0.0%`**。

---

## 5. 面向解决方案架构师的生产落地清单

1. **SHARP 树配额治理（Quota Admission Control）**：绝不允许在多租户 Slurm/Kubernetes 集群中让所有小型调试作业或次要并行组（如小消息状态同步）无限制抢占 `NCCL_COLLNET_ENABLE=1`。应通过调度器 Admission Webhook 仅将硬件 SHARP Group 授予跨机规模 $\ge 8$ 节点的 DP/FSDP/CP 主通信域。
2. **回退算法护栏**：当 SHARP 树满载时，确保跨节点 communicator 优先回退至 `NCCL_ALGO_NVLS_TREE`（`7.557 ms`），并通过 RoCE DSCP / InfiniBand SL 隔离多播与单播队列，严防退化至大环路 `Ring`（`15.875 ms`）引发全局拥塞扩散。
3. **可复现性审计开关**：在执行 SDC（Silent Data Corruption）梯度哈希对账或 RL 策略精确复现时，若发现跨节点 AllReduce 哈希不一致，优先检查是否混用了 `CollNet` 动态树与主机端 `Tree/Ring` 归约。

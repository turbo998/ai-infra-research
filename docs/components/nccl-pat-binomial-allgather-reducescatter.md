# NCCL PAT (`NCCL_ALGO_PAT`) Parallel Aggregated Trees 二项树集合通信架构

> **证据分类**：`implementation`（基于 `NVIDIA/nccl` `commit 3e46c089f4ae4732722a419eb6096aead7e47a30` 一手源码核验）+ `own-offline`（`experiments/nccl_pat_binomial_allgather_rs/run.py` 确定性步进与延迟/带宽仿真）。本文不宣称已完成千卡物理 RDMA 集群实测剖析（`own-hardware`）。

---

## 1. 为什么传统 Ring `AllGather` / `ReduceScatter` 在千卡规模遭遇延迟墙？

在 FSDP2 / ZeRO-3 训练中，每一层前向计算前需执行 `AllGather` 收集分片参数，反向计算后需执行 `ReduceScatter` 归约分片梯度。
- 传统 `NCCL_ALGO_RING` 将 $N$ 个 Rank 组织为逻辑环，每次集合通信必须严格经历 **$N - 1$ 个串行步（Sequential Steps）**。当集群规模扩展至 $N = 1,024$ 卡（128 台 8 卡节点）时，即使单跳 RDMA 延迟仅 $1.85\ \mu\text{s}$、NVLink 延迟仅 $0.45\ \mu\text{s}$，仅环路流水线填充与排空（Pipeline Fill/Drain）就高达 **`2.1+ ms`**；
- 更致命的是，对于中小型参数张量（如 $M = 4\text{ MiB} \sim 16\text{ MiB}$ 的 LayerNorm、Router 或单层 Attention 投影），Ring 算法将消息切分为 $N \times C$（$C=16$ 个 Channel）块，单步单 NIC 发送载荷骤降至 **`256 B ~ 1 KiB`**，落入 RDMA WQE 头部与 PCIe 事务开销主导的低效区，导致有效总线带宽（Bus Bandwidth）崩塌至不足峰值带宽的 `15%`。
- 虽然 `NCCL_ALGO_TREE`（双二叉树）将 `AllReduce` 延迟降为 $O(\log_2 N)$，但传统 Tree 算法依赖归约与广播对称流水线，无法直接用于单独的 `AllGather` 或 `ReduceScatter`。为此，NCCL 2.23+ 引入了专为大规模 `AllGather` / `ReduceScatter` 设计的 **`NCCL_ALGO_PAT`（Parallel Aggregated Trees，并行聚合二项树）**。

---

## 2. 一手源码核验：`PatRSAlgorithm`、`PatAGAlgorithm` 与 Multi-RPN NVLS 协同

我们对 `NVIDIA/nccl`（`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`）中的 3 个核心文件进行了源码级核验：

### 2.1 动态消息聚合因子（`aggFactor`）与位反转二项树：`src/include/collectives.h`（blob `f0dac7b9758627fcecde8c989e4f63f7e2a4a66d`，第 454–942 行）
在 `PatRSAlgorithm<T>`（第 537–560 行）与 `PatAGAlgorithm<T>`（第 777–800 行）构造函数中，PAT 根据单 Channel 数据量 `channelSize * sizeof(T)` 与 FIFO 步长缓冲 `stepSize` 动态计算聚合因子 `aggFactor`：
```cpp
aggDelta = nrPow2 = (1 << log2Up(nranks));
aggFactor = 1;
size_t channelSize = end - offset;
while (stepSize / (channelSize * sizeof(T) * aggFactor) >= 2 && aggFactor < nranks / 2) {
  aggFactor *= 2;
  aggDelta /= 2;
}
postFreq = aggFactor;
if (postFreq < parallelFactor) parallelFactor = postFreq;
int d = stepDepth;
while (d > 1 && aggFactor < nranks / 2) {
  d /= 2;
  aggFactor *= 2;
  aggDelta /= 2;
}
```
- **核心机制 1：小消息硬件级聚合（`postFreq` & `aggFactor`）**：当 `nranks` 很大导致单 Rank 切片 `channelSize * sizeof(T)` 很小时，`aggFactor` 自动按 2 的幂次倍增（最高可达 `nranks / 2`），将原本分散在多个通信步的细碎切片聚合进同一个 FIFO 槽位后再一次性触发 `ps->postSend = 1` / `ps->postRecv = 1`（第 586–604 行），彻底消除小包 RDMA Doorbell 风暴。
- **核心机制 2：`mirrorInvert` 与 `firstBitSet` 超立方体（Hypercube）无冲突调度（第 485–501 行）**：
  PAT 使用 `mirrorInvert(a, lastA)` 生成位反转序列（例如在 `max=16` 时生成 `15, 7, 11, 3, 13, 5, 9, 1, ...`），并通过 `firstBitSet(s, nrPow2)`（底层调用 CUDA PTX `__ffs(i)`）计算每一步在二项树超立方体上的收发维度 `recvDim` 与 `sendDim`（最多支持 `NCCL_PAT_NDIMS = 32` 维，覆盖 $2^{32}$ Ranks）。

### 2.2 专用调度线程与多 Warp 并行工作组：`src/device/all_gather.h`（blob `a9ad0e19cd073f8043f3cd0813276e2a5b2dee84`）与 `src/device/reduce_scatter.h`（blob `04150df82c7d76a54b8ef90029055792598e1426`）
在 `RunWorkColl<..., NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE>` 中，CTA 内部采用 **Warp-Specialized 生产者-消费者架构**：
1. **单线程算法调度器（Algo Computation Thread）**：由 `tid == nworkers`（One-RPN）或 `tid == NCCL_MAX_NTHREADS - 1`（Multi-RPN）专门执行 `patAlgo.getNextOp(ps)`，将计算出的 `ncclPatStep` 描述符写入长度为 `NCCL_SHMEM_PAT_STEPS = 32` 的共享内存环形队列 `shmem->patSteps`，并通过 `cuda::atomic_ref<int, cuda::thread_scope_block>` 发布步进状态；
2. **`parallelFactor` 组并行通信 Worker Warps**：前 `nworkers` 个线程按 `parallelFactor` 划分为 `nGroups` 个独立 Warp 组，分别并行消费 `step = group + k * nGroups` 的二项树传输任务（调用 `prims.patCopy` 或 `prims.patReduce`）；
3. **Multi-RPN（每节点多卡）下的 NVLS + 跨节点 PAT 两级混合架构（`all_gather.h:189-268`, `reduce_scatter.h:154-227`）**：
   当每节点包含 `localRanks = 8` 张 GPU（`work->isOneRPN == false`）时，PAT 将算法实例化的规模从总卡数 `nranks` 压缩为节点数 `nNodes = nranks / 8`，并将 `patCount` 设为 `count * localRanks`；同时分配专门的 `nGatherWorkers` / `nScatterWorkers` 线程组通过 `FanAsymmetric<NCCL_MAX_NVLS_ARITY>` 驱动节点内 NVSwitch (`NVLS`) 硬件多播/归约（`prims.patGather` / `prims.patScatter`），使跨节点网络跳数进一步从 $\log_2(N)$ 降至 $\log_2(N / 8)$！

---

## 3. 原创 CPU/Offline 实验与定量对比

实验脚本：`experiments/nccl_pat_binomial_allgather_rs/run.py`（结果固化于 `experiments/nccl_pat_binomial_allgather_rs/results.json`）。

### 3.1 跨集群规模（64 ~ 1,024 卡）与消息尺寸（1 MiB ~ 256 MiB）评测矩阵

| 集群规模 $N$ (Nodes) | 消息大小 (MiB) | Ring 串行跳数 | PAT (Multi-RPN) 跨节点跳数 | PAT (1-RPN / Multi-RPN) `aggFactor` | Ring 延迟 ($\mu\text{s}$) | PAT (1-RPN) 延迟 ($\mu\text{s}$) | PAT (Multi-RPN+NVLS) 延迟 ($\mu\text{s}$) | Ring 总线带宽 (GB/s) | PAT (Multi-RPN) 总线带宽 (GB/s) | 端到端加速比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **64 (8N)** | 4 MiB | 63 | **3** | `32 / 4` | 227.70 | 109.92 | **107.15** | 18.132 | **38.533** | **2.13x** |
| **256 (32N)** | 4 MiB | 255 | **5** | `128 / 16` | 628.41 | 114.74 | **111.98** | 6.648 | **37.311** | **5.61x** |
| **512 (64N)** | 4 MiB | 511 | **6** | `256 / 32` | 1161.42 | 116.78 | **114.01** | 3.604 | **36.716** | **10.19x** |
| **1,024 (128N)** | 1 MiB | 1,023 | **7** | `512 / 64` | 2157.36 | 49.46 | **45.28** | 0.485 | **23.141** | **47.64x** |
| **1,024 (128N)** | 4 MiB | 1,023 | **7** | `512 / 64` | 2227.17 | 118.73 | **115.96** | 1.882 | **36.143** | **19.21x** |
| **1,024 (128N)** | 16 MiB | 1,023 | **7** | `512 / 64` | 2506.42 | 398.17 | **401.04** | 6.687 | **41.803** | **6.25x** |
| **1,024 (128N)** | 64 MiB | 1,023 | **7** | `512 / 64` | 3623.40 | 1524.22 | **1549.68** | 18.500 | **43.254** | **2.34x** |
| **1,024 (128N)** | 256 MiB | 1,023 | **7** | `256 / 32` | 8091.10 | 6029.66 | **6145.46** | 33.143 | **43.638** | **1.32x** |

---

## 4. 关键架构师结论与反证边界

1. **彻底攻克 FSDP2 千卡中小型分片集合通信墙**：在 `1,024` 卡规模下，对于 `4 MiB` 与 `16 MiB` 的 `AllGather`/`ReduceScatter`，Multi-RPN PAT + NVLS 将串行网络跳数从 `1,023` 步压缩至 **`7` 步**，通信延迟分别从 `2,227.17 us` / `2,506.42 us` 降至 **`115.96 us`（提速 `19.21x`）** 与 **`401.04 us`（提速 `6.25x`）**，有效总线带宽从 `6.687 GB/s` 跃升至 **`41.803 GB/s`**。
2. **何时 Ring 仍优于或持平 PAT（反证边界）？**
   - 当集群规模较小（如 $N \le 64$）且单次集合通信消息极大（如 $M \ge 256\text{ MiB}$ 聚合 Bucket）时，Ring 的单步分片已完全填满 512 KiB FIFO 槽位，且 $63$ 步的延迟占比不足 `3%`；此时 Multi-RPN PAT 因额外经过一次节点内 NVLS Gather/Scatter 暂存，在 `N=64, M=256 MiB` 下延迟为 `6,049.0 us`，略逊于 Ring 的 `6,007.98 us`（`0.99x`）。

---

## 5. 架构师练习与可证伪问题

1. **计算练习**：根据 `PatRSAlgorithm` 构造函数公式，当 `stepSize = 524,288 B`、`stepDepth = 8`、`nranks = 256`、`channelSize * sizeof(T) = 8,192 B` 时，手算第一个 `while` 循环结束时的 `postFreq` 以及第二个 `while` 循环结束时的最终 `aggFactor` 和 `aggDelta`。
2. **可证伪假说**：在 FSDP2 逐层 `DTensor` 调度中，若单层参数量 $< 32\text{ MiB}$ 且 DP 域规模 $\ge 256$，开启 `NCCL_ALGO=PAT` 相比强制 `NCCL_ALGO=RING` 可将难以被前向计算掩盖的暴露通信时间削减 `70%` 以上。

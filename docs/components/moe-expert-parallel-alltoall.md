# 大规模 MoE 专家并行（EP）与 DeepEP 双层内核架构（NVLink/RDMA 协同去重、EPLB 冗余专家预取与负载倾斜消解）

面向解决方案架构师（Solution Architect），本篇组件文档基于 **DeepEP (`deepseek-ai/DeepEP` commit `b5a33eefda39485f438dc764695b71ba7b9249b6`)** 的一手 CUDA/C++ 源码，深入剖析 `DeepSeek-V3` / `Qwen3-MoE` / `Mixtral` 等大规模混合专家模型（MoE）在 Expert Parallelism (EP) 下的 Token 数据包对齐、两级 `Scale-Out (RDMA) + Scale-Up (NVLink)` 协同去重转发、动态寄存器重分配（`warpgroup_reg_realloc`）、专家负载倾斜（Zipfian Skew）下的冗余专家在线权重预取（`EPLB`），并配套可复现标准库实验 `experiments/moe_ep_alltoall/run.py`（`own-offline`）。

---

## 1. 工程问题定义：为什么细粒度 MoE 会击穿传统扁平 All-to-All？

在 `Mixtral-8x22B`（`8` 专家，`top-2`）时代，专家数少且每个 Token 仅激活 2 个专家。而以 `DeepSeek-V3`（`256` 个路由专家，每 Token 激活 `top-8` 路由专家 + 1 共享专家，`hidden = 7168`）为代表的现代细粒度 MoE（Fine-Grained MoE）引入了三大严峻的通信与负载均衡挑战：

1. **跨节点重复发送放大（Cross-Node Fanout Amplification）**：当 `256` 个专家分布在 `8` 台 8-GPU 节点（`EP=64`，每 GPU `4` 个专家，每节点 `32` 个专家）时，一个 Token 选出的 `top-8` 专家极大概率有多个专家落在同一个远端节点的不同 GPU 上。若采用传统扁平（Flat）`ncclAllToAll`，源 GPU 会向该远端节点的不同 GPU 重复发送多份完全相同的 Token 激活向量，浪费宝贵的跨节点 RDMA 带宽（`400GbE` 仅约 `48.5 GB/s`，远低于节点内 NVLink 的 `400–900 GB/s`）。
2. **FP8 Dispatch 与 BF16 Combine 的非对称硬件对齐与 SM 共享内存（SMEM）挤兑**：Dispatch 阶段可传输 FP8 激活值 + `sf_pack_t`（FP32 或 `ue8m0x4` 块缩放因子）+ 路由元数据；而 Combine 阶段需对 `top-8` 专家输出做加权归约，通常传输 `nv_bfloat16`（体积翻倍）。如何在单个持久化 CUDA Kernel 内同时容纳 Notify Warps、Scale-Out RDMA Warps 与 Scale-Up NVLink Forwarding Warps 而不撑爆 Hopper/Blackwell SM 的 `227 KiB`（`232,448 B`）共享内存与寄存器堆？
3. **Zipf 专家负载倾斜与 GroupGEMM 对齐填充（`kExpertAlignment`）**：真实推理与训练中热门专家被高频选中，且 Tensor Core FP8 GroupGEMM 要求每个专家的接收 Token 数向上对齐到 `kExpertAlignment`（如 `128` tokens），导致木桶效应（Straggler Rank）拖慢整个 EP 同步步。

```
+---------------------------------------------------------------------------------------------------+
|                  DeepEP 2-Tier Hybrid Dispatch (`hybrid_dispatch.cuh`) Pipeline                   |
|                                                                                                   |
|  [Source Rank (Scale-out s_src, Scale-up u_src)]                                                  |
|   Token x (7168B FP8) + SF (224B) + Metadata (100B->128B) ===> 7,552 B RDMA Packet                |
|     |                                                                                             |
|     |-- 1. Notify Warps (4 warps):                                                                |
|     |      `ptx::deduplicate(dst_rank_idx)` + 64-bit `(1ULL << 32) | count` Grid Reduction        |
|     |      SM 0 exchanges counts via `gin.put<ncclTeamTagRail>` & aligns to `kExpertAlignment`    |
|     |                                                                                             |
|     |-- 2. Scale-out Send Warps (`kNumScaleoutWarps = 8`):                                        |
|     |      `ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx)`                             |
|     |      Sends AT MOST 1 RDMA copy per target Scale-out Node via `gin.put<ncclTeamTagRail>`!    |
|     |      Batches tail notifications every `kScaleoutUpdateInterval = 6` tokens                  |
|     v                                                                                             |
|  [Target Intermediate Rank (Scale-out s_dst, same Scale-up rail u_src)]                           |
|     |-- 3. Forward Warps (`kNumForwardWarps = 8`):                                                |
|            Polls `scaleout_channel_signaled_tail` in chunks of `kNumSlotsPerForwardChunk = 6`     |
|            `ptx::tma_load_1d` into SMEM -> deduplicates target local NVLink ranks (`u_dst`)       |
|            -> `ptx::tma_store_1d` over NVLink (`ncclTeamTagLsa`) to final expert ranks!           |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. 一手源码深挖：`deepseek-ai/DeepEP` 核心数据结构与内核实现

### 2.1 `TokenLayout` 与硬件双层对齐算术（`deep_ep/include/deep_ep/layout/ep/token.cuh` & `compiled.cuh`）

在 `compiled.cuh:59-71` 与 `token.cuh:10-41` 中，DeepEP 定义了严格的硬件对齐常数与单 Token 封包布局：
- **硬件对齐常数**：
  - `kNumTMAAlignmentBytes = 32`（Hopper/Blackwell TMA 1D 异步拷贝对齐要求）
  - `kNumRDMAAlignmentBytes = 64`（PCIe/RDMA 缓存行对齐要求）
  - `kNumAllocationAlignmentBytes = 2 * 1024 * 1024`（`2 MiB` 巨页对称窗口对齐）
  - `union sf_pack_t { float fp32; int ue8m0x4; }`（`4 B`，支持每 128 元素 1 个 FP32 缩放因子，或 Blackwell MXFP8 `ue8m0x4` 4 合 1 指数打包）
- **`TokenLayout` 精确字节公式（`token.cuh:18-41`）**：
  $$\text{num\_metadata\_bytes} = k \cdot (\text{sizeof(int)} + \text{sizeof(float)}) + \begin{cases} (1 + k)\cdot \text{sizeof(int)}, & \text{if with\_metadata} \\ 0, & \text{otherwise} \end{cases}$$
  $$\text{num\_data\_bytes} = \text{align}_{64}\Big(\text{align}_{32}(\text{num\_hidden\_bytes}) + \text{align}_{32}(\text{num\_sf\_bytes}) + \text{align}_{32}(\text{num\_metadata\_bytes})\Big)$$
  $$\text{get\_num\_bytes}\langle\text{kWithMBarrier}\rangle = \text{num\_data\_bytes} + \text{align}_{32}\big(\text{kWithMBarrier} ? \text{sizeof(mbarrier)} : 0\big)$$
  - 以 **`DeepSeek-V3` (`hidden = 7168`, `num_sf_packs = 56`, `k = 8`)** 为例：
    - **Dispatch 阶段（`dispatch.hpp:26-29`，`with_metadata = true`）**：
      - `num_hidden_bytes = 7,168 B`（FP8），`num_sf_bytes = 56 * 4 = 224 B`（均已 32B 对齐）。
      - `num_metadata_bytes = 8 * 8 + (1 + 8) * 4 = 100 B`，经 $\text{align}_{32}(100) = 128\text{ B}$。
      - 三者之和为 $7168 + 224 + 128 = 7,520\text{ B}$，再经 RDMA 64B 对齐得 **$\text{num\_data\_bytes} = 7,552\text{ B}$**！
      - 在 SMEM TMA 槽位中附加 `mbarrier`（$\text{align}_{32}(8) = 32\text{ B}$），单 Warp SMEM 槽位精确为 **`7,584 B`**。
    - **Combine 阶段（`combine.hpp:19-22`，`nv_bfloat16`，`num_sf_bytes = 0`，`with_metadata = false`）**：
      - `num_hidden_bytes = 7168 * 2 = 14,336 B`，`num_metadata_bytes = 8 * 8 = 64 B`。
      - RDMA 传输包大小为 **`14,400 B`**，单 Warp SMEM 槽位为 **`14,432 B`**。
      - 当 `num_channels_per_sm = 8` 时，`hybrid_combine` 每个 SM 启动 `8` 个 Scale-Up Warps + `8` 个 Forward Warps = `16` 个 Warps（`combine.hpp:54-57`），占用 SMEM $16 \times 14,432 = \mathbf{230,912\text{ B}}$（`225.5 KiB`），恰巧卡在 Hopper SM 最大共享内存（`232,448 B = 227 KiB`）之内，余量仅 **`1,536 B`**！

### 2.2 `hybrid_dispatch.cuh` 与 `hybrid_combine.cuh`：两级去重、流水步进与跨 Warpgroup 寄存器借用

在 `deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh`（`674` 行）与 `hybrid_combine.cuh`（`627` 行）中，DeepEP 将同一个 CTA（Block）内的线程划分为不同职能的 Warp 角色：

1. **64 位打包计数器与单阶段网格归约（`hybrid_dispatch.cuh:140-171`）**：
   - Notify Warps 先在块内共享内存用 `atomicAdd_block` 统计发往各 Rank 和各 Expert 的 Token 数，随后每个 SM 通过一行原子加 `ptx::red_add(notify_reduction_ptr + i, (1ll << 32ll) | rank_expert_count[i])` 同时累加**高 32 位到达 SM 计数**与**低 32 位 Token 计数**。
   - `SM 0` 只需轮询 `(status >> 32) == kNumSMs`，即可在不引入全网格 Barrier 的情况下立刻获知所有 SM 统计完毕，并由 `SM 0` 统一对齐到 `kExpertAlignment`（`hybrid_dispatch.cuh:269-270`）：
     `const auto aligned_count = math::align<int>(count, thread_idx < kNumScaleupRanks ? 1 : kExpertAlignment);`
2. **两级 `ptx::deduplicate` 硬件内联去重（`hybrid_dispatch.cuh:395, 412, 584`）**：
   - 在 Scale-Out 发送 Warp 中，先将目标专家编号 `dst_expert_idx` 映射为目标节点号 `stored_dst_scaleout_rank_idx = dst_expert_idx / kNumExpertsPerScaleout`，然后调用 `ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx)`。
   - **核心收益**：即便一个 Token 的 `top-8` 专家中有 3 个落在远端节点 $s$ 的不同 GPU 上，Scale-Out Warp 也**只向节点 $s$ 发送 1 次 IBGDA/NCCL Gin RDMA Put**（`gin.put<ncclTeamTagRail>`），并且当该 Token 的所有目标都在本节点时直接跳过 `scaleout_send_buffer` 的 TMA 写入（`if (scaleout_rank_mask ^ (1 << scaleout_rank_idx))` at line `425`）。
   - 当数据包到达目标节点 $s$ 后，Forward Warp 从 `scaleout_recv_buffer` 通过 TMA 读入 SMEM，再次执行节点内 NVLink 去重 `ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx)`（line `584`），通过对称 NVLink 窗口 `gin.get_sym_ptr<ncclTeamTagLsa>` 执行 `ptx::tma_store_1d` 扇出到节点内对应的目標 GPU。
3. **批量尾指针通知（`kScaleoutUpdateInterval = 6` / `kNumScaleupUpdateInterval = 3`）与非对称寄存器重分配（`hybrid_combine.cuh:102-113`）**：
   - 为避免每发 1 个 Token 就执行昂贵的跨节点 RDMA 原子尾指针更新或全系统 `st.release.sys`，`hybrid_dispatch.cuh:27, 340` 每满 `6` 个槽位才批量合并发送一次 `gin.red_add_rel<ncclTeamTagRail>`，Forward Warp 亦以 `kNumSlotsPerForwardChunk = 6` 为粒度轮询搬运（line `529`）。
   - 在 `hybrid_combine.cuh:102-113` 中，由于 Scale-Up 发送 Warp 只负责发起 TMA Load/Store，而 Forward Warp 需要在寄存器中对多个专家的 `nv_bfloat16` 向量做加权累加归约，DeepEP 使用 Hopper WGMMA 硬件指令 `ptx::warpgroup_reg_realloc` 将 Scale-Up Warps 的每线程寄存器主动压缩至 **`40` 个**，从而将 **`216` 个寄存器/线程**（`256 - 40`）让渡给 Forward Warps！

### 2.3 `eplb.cuh` 与 `prefetch_weights.cuh`：冗余专家在线负载均衡（EPLB）

在 `deep_ep/include/deep_ep/layout/ep/eplb.cuh`（`143` 行）与 `deep_ep/include/deep_ep/impls/ep/prefetch_weights.cuh`（`95` 行）中，DeepEP 原生提供了无需中断推理/训练的冗余专家权重动态同步机制：
- **任务构建 `EPLBSharedMemoryLayout::build_tasks`（`eplb.cuh:29-64`）**：
  - 给定长度为 `kNumRanks * kNumRedundantExperts` 的全局映射表 `redundancy_mapping`（记录每个 Rank 的每个冗余专家槽位当前应当镜像哪一个全局热门专家 `expert_idx`），每个源 Rank 在 `32 KiB` 共享内存（最多容纳 `kNumMaxTasks = 2,730` 个 `EPLBTask`）中用 `cub::BlockScan::ExclusiveSum` 筛选出属于本 Rank 持有的源专家（`expert_idx / kNumLocalExperts == rank_idx`），生成交换三元组 `{local_expert_idx, dst_nvl_rank_idx, redundant_expert_idx}`。
- **无锁细粒度块认领 `EPWeightIterator`（`eplb.cuh:91-136` & `prefetch_weights.cuh:71-86`）**：
  - 将最多 `kNumMaxWeightEntries = 16` 个权重张量按固定块大小 `kNumBytesPerChunk`（如 `16 KiB`）切分，所有 SM 的所有 Warp 通过全局原子计数器 `atomicAdd(chunk_idx_counter, 1)` 动态认领 `(weight_idx, chunk_offset, task_idx)`，使用 `tma_load_1d` 读入本地 SMEM 后立即通过 `tma_store_1d` 直写远端 NVLink Rank 的 `redundant_expert_weights` 对称地址，彻底消除长尾源 Rank 的拷贝木桶效应。

---

## 3. 原创 CPU/Offline 实验结果与反例分析（`experiments/moe_ep_alltoall/run.py`）

### 3.1 Token 封包对齐与 SM 共享内存容量边界

结果摘自 `experiments/moe_ep_alltoall/results.json`（`own-offline`）：

| Case ID | 模型配置 | Dispatch 单 Token RDMA 包 (`B`) | Dispatch 单 Warp SMEM (`B`) | Combine 单 Token RDMA 包 (`B`) | Combine 单 Warp SMEM (`B`) | Notify SMEM (`B`) | Hybrid Dispatch SMEM (`B`) | Hybrid Combine SMEM (`B`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `deepseek_v3_h7168_fp8_topk8_ep64` | DeepSeek-V3 (`H=7168`, FP8, `k=8`, `EP=64`) | **`7,552 B`** | `7,584 B` | **`14,400 B`** | `14,432 B` | `1,536 B` | `122,880 B` | **`230,912 B`** (`99.3%` of `227 KiB`) |
| `qwen3_moe_h4096_fp8_topk8_ep32` | Qwen3-235B-A22B (`H=4096`, FP8, `k=8`, `EP=32`) | **`4,352 B`** | `4,384 B` | **`8,256 B`** | `8,288 B` | `1,024 B` | `71,168 B` | `132,608 B` |
| `mixtral_8x22b_h6144_bf16_topk2_ep8` | Mixtral-8x22B (`H=6144`, BF16, `k=2`, `EP=8`) | **`12,352 B`** | `12,384 B` | **`12,352 B`** | `12,384 B` | `512 B` | `99,584 B` | `99,072 B` |

### 3.2 两级 `Scale-Out + Scale-Up` 去重 vs 扁平 All-to-All

针对 `4,096` tokens/rank 的批次，对比扁平 RDMA 发送与 DeepEP `ptx::deduplicate` 节点级去重 + 节点内 NVLink 转发流水线：

| Case ID | 拓扑 (`ScaleOut x ScaleUp`) | 节点路由限制 (`G`) | 扁平每 Token 远端 RDMA 次数 | DeepEP 去重后远端 RDMA 次数 | 扁平 RDMA 流量/Rank (`MiB`) | DeepEP RDMA 流量/Rank (`MiB`) | RDMA 流量削减比 | 扁平 RDMA 耗时 (`us`) | DeepEP 流水总耗时 (`us`) | 端到端加速比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dsv3_ep64_8nodes_x_8gpus_group_limited_4nodes` | `8 x 8` (`H800/H100`) | `4` 节点 (`DeepSeek-V3` 组受限路由) | `7.0000` | **`3.1754`** | `600.250 MiB` | **`272.289 MiB`** | **`2.20x`** | `12,977.48 us` | **`6,136.54 us`** | **`2.11x`** |
| `dsv3_ep64_8nodes_x_8gpus_unrestricted_8nodes` | `8 x 8` (`H800/H100`) | `8` 节点 (无限制随机路由) | `7.0000` | **`4.6328`** | `600.250 MiB` | **`397.264 MiB`** | **`1.51x`** | `12,977.48 us` | **`8,849.85 us`** | **`1.47x`** |
| `gb200_ep64_2racks_x_32gpus_hybrid` | `2 x 32` (`GB200` 跨机柜) | `2` 机柜 (`32` GPU/域) | `4.0000` | **`0.9965`** | `343.000 MiB` | **`85.451 MiB`** | **`4.01x`** | `3,707.85 us` | **`1,052.38 us`** | **`3.52x`** |

**关键工程发现**：
1. **`DeepSeek-V3` 的节点组受限路由（Node-Limited Routing，限制每 Token 最多命中 `4` 个节点）与 DeepEP `ptx::deduplicate` 是软硬协同的一体两面**：若不限制路由节点数（`G = 8`），`top-8` 平均散落到 `5.29` 个节点（远端 `4.63` 个），去重比仅 `1.51x`；配合 `G = 4` 组受限路由后，去重比跃升至 **`2.20x`**，单 Rank 跨节点 RDMA 流量从 `600.25 MiB` 降至 `272.29 MiB`。
2. **在 `GB200` 大 NVLink 域（`2 x 32`）下，两级去重将跨机柜 RDMA 流量削减 `4.01x`**：因为单个 Scale-Up 域涵盖 `128` 个专家，任何落入远端机柜的複数个专家在跨机柜 RDMA 层都被折叠为 **至多 `1` 次传输**（期望 `0.9965` 次/Token）。

### 3.3 EPLB 冗余专家负载均衡收益与“小批次过度复制”反例

在 `EP=32`（`256` 专家，每 Rank `8` 个原始专家，每个专家权重 `42 MiB = 44,040,192 B`）下测试不同 Zipf 倾斜度 $s$ 与冗余槽位数：

| Case ID | Zipf 倾斜指数 $s$ | 集群总 Token 数 | 冗余专家槽位/Rank | 无 EPLB 最慢 Rank Token 数 (不均衡度) | 启用 EPLB 最慢 Rank Token 数 (不均衡度) | 最热源 Rank 权重预取耗时 (`ms`) | 无 EPLB 步耗时 (`ms`) | 启用 EPLB 步总耗时 (`ms`) | 端到端加速比 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ep32_256experts_zipf_0_85_with_1_redundant_per_rank` | `0.85` | `131,072` | `1` (`+12.5%` 显存) | `351,872` (`10.58x`) | **`86,912`** (`2.61x`) | `2.863 ms` (`26` 任务) | `19.548 ms` | **`7.691 ms`** | **`2.54x`** |
| `ep32_256experts_zipf_1_15_heavy_skew_2_redundant` | `1.15` | `131,072` | `2` (`+25.0%` 显存) | `583,040` (`17.52x`) | **`80,256`** (`2.40x`) | `6.166 ms` (`56` 任务) | `32.391 ms` | **`10.624 ms`** | **`3.05x`** |
| `ep32_256experts_microbatch_over_replicated_counterexample` (**反例**) | `0.65` | `4,096` (小批次) | `4` (`+50.0%` 显存) | `7,552` (`5.02x`) | `1,536` (`1.00x`) | **`5.835 ms`** (`53` 任务) | **`0.420 ms`** | **`5.921 ms`** | **`0.07x` (严重劣化!)** |

- **反例与架构护栏（为什么不能在每个推理 Decode 微批次都全量触发 `lb_prefetch_weights`？）**：
  在 `ep32_256experts_microbatch_over_replicated_counterexample` 中，集群总 Token 数仅 `4,096`（对应低并发 Decode 微批次），即使不均衡度为 `5.02x`，最慢 Rank 的专家计算耗时也仅 **`0.420 ms`**。若此时激进重排并同步拷贝 `53` 个冗余专家副本（`53 * 42 MiB = 2.23 GiB` 权重），即便走 `3,200 Gbps`（`400 GB/s`）NVLink TMA 也要耗费 **`5.835 ms`**，导致单步耗时暴涨 **`14.1x`**（加速比 `0.07x`）！
  **结论**：在线 EPLB 权重重排（`lb_prefetch_weights`）必须采用**低频周期性触发（如每数百步或基于滑动窗口统计异步双缓冲更新）**，或仅在大吞吐 Prefill/训练批次（计算收益 $\Delta T_{\text{compute}} \gg T_{\text{prefetch}}$）中同步执行。

---

## 4. 证据等级与已知边界

- **`implementation`（已核验一手源码）**：
  - `deepseek-ai/DeepEP` (`b5a33eefda39485f438dc764695b71ba7b9249b6`)：`compiled.cuh`、`token.cuh`、`workspace.cuh`、`eplb.cuh`、`ep.hpp`、`dispatch.hpp`、`combine.hpp`、`dispatch.cuh`、`hybrid_dispatch.cuh`、`hybrid_combine.cuh`、`prefetch_weights.cuh`。
- **`own-offline`（本项目可复现合成实验）**：
  - `experiments/moe_ep_alltoall/run.py` 与 `tests/test_moe_ep_alltoall.py`。
- **边界说明**：本实验严格复现 `TokenLayout` 字节对齐公式与两级去重/EPLB 算术模型，不声称在真实 64 卡 H800/GB200 物理集群上采集了实测 IBGDA 延迟。

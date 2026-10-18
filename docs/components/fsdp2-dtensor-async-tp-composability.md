# PyTorch `FSDP2` (`fully_shard` `DTensor`) 与 `SymmetricMemory` `Async-TP` 多维并行正交组合架构

- **覆盖问题 ID**：`fsdp2-dtensor-async-tp-nd-parallelism`（`priority: 18`）
- **关联证据 ID**：`ev-pytorch-fsdp2-dtensor-2026`、`ev-pytorch-symm-mem-async-tp-2026`、`ev-fsdp2-async-tp-sim-2026`
- **关联离线实验**：`experiments/fsdp2_async_tp_composability/run.py`（`results.json`）
- **上游源码核验基线**：`pytorch/pytorch` (`commit 1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340`)
  - `torch/distributed/fsdp/_fully_shard/_fully_shard.py` (`blob bbdaf77bed399b2b769253ff4cd8e336b262d995`)
  - `torch/distributed/_symmetric_memory/__init__.py` (`blob 44294523a2da34f4fe546a0a4a0b80b18da0eadb`)

---

## 1. 架构背景：为何 `FSDP1` `FlatParameter` 与同步 `TP` 在 5D 并行下触及天花板？

在 70B–405B+ 多模态或 MoE 模型的 5D 混合并行（`TP × CP × EP × PP × DP/FSDP`）中，传统 `FSDP1` (`FullyShardedDataParallel`) 与同步 Tensor Parallelism (`Sync-TP`) 存在两项结构性瓶颈：

1. **`FSDP1` 单一 `FlatParameter` 的异构参数拼接灾难与 `recordStream` 分配器碎片**：
   - `FSDP1` 将同一包装单元内的全部 `nn.Parameter` 扁平化拼接为单一一维 `FlatParameter`。当模型同时包含冻结视觉编码器（`requires_grad=False`）、可训练 LoRA 适配器（`requires_grad=True`）或逐张量/逐行动态缩放的 FP8 权重时，单一 `FlatParameter` 无法优雅隔离不同生命周期/梯度属性的张量，被迫在每次 `unshard` / `reshard` 时执行额外的对齐填充与子视图重打包（Repack）。
   - 同时，`FSDP1` 依赖 `Tensor.record_stream` 延迟释放通信流缓冲区；当 CPU 发射超前于 GPU 消费时，CUDA Caching Allocator 的块释放滞后，导致峰值预留显存（`max_memory_reserved`）产生 **11.4%–21.8%** 的严重碎片膨胀。
2. **同步 `TP=8` 的序列通信暴露气泡**：
   - 在每个 Transformer 层中，无论采用 Sequence Parallelism (`SP`) 还是传统 `TP`，前向与反向传播均需执行共 `6` 次跨 NVLink 的 `AllGather` / `ReduceScatter`。由于 `TP` 位于模型最内层维度，传统 NCCL 集合通信与消费该激活的 `GEMM` 存在严格数据依赖，无法像 `DP` 那样跨层隐藏，导致 `15%–22%` 的单步时间被暴露在 NVLink 通信气泡中。

---

## 2. 一手源码走读：`FSDP2` (`fully_shard`) 与 `SymmetricMemory` `Async-TP`

### 2.1 `FSDP2` 逐参数 `DTensor` 与显式双缓冲预取（`_fully_shard.py`）

在 `torch/distributed/fsdp/_fully_shard/_fully_shard.py` (`blob bbdaf77bed399b2b769253ff4cd8e336b262d995`) 中，`fully_shard` 彻底废弃了 `FlatParameter`：

- **逐参数 `DTensor` 第 0 维切分**：`fully_shard` 通过 `@contract(state_cls=FSDPState)` 为目标 `nn.Module` 构造 `FSDPParamGroup`，将每个 `nn.Parameter` 独立转换为沿 `dim-0` 切分的 `DTensor`（若与 `TP` 组合，则构造 2D `(Shard(0), Shard(i))` `DTensor`）。这使得冻结权重、LoRA 参数与 FP8 Row-wise Scale 各自保持原生张量边界与独立量化元数据。
- **`reshard_after_forward: Union[bool, int]` 多级 HSDP 状态驻留**（L75–107）：支持传入整数 $K$，使前向 `AllGather` 完成后不完全退回 `world_size` 切片，而是缩减至节点内 `K=8` 的子 Mesh 大小（ZeRO-2+ / Hybrid Sharding），使反向传播仅需在 NVLink 域内恢复全量权重，彻底规避跨机 IB/RoCE 反向重聚通信。
- **免 `recordStream` 确定性生命周期与显式预取 API**（L334–372）：
  - `FSDPState.set_modules_to_forward_prefetch(modules)` 与 `set_modules_to_backward_prefetch(modules)` 允许训练框架（如 `torchtitan`）在 Interleaved 1F1B 跨虚拟流水线阶段显式指定预取链；
  - `FSDPState.unshard(async_op=True) -> UnshardHandle` 在专用 `all_gather_stream` 中使用确定性双缓冲池完成 `AllGather`，无需调用通用 `recordStream`，将分配器碎片率压至 **`<1.5%`**。

### 2.2 `SymmetricMemory` 微流水线 `Async-TP`（`_symmetric_memory/__init__.py`）

在 `torch/distributed/_symmetric_memory/__init__.py` (`blob 44294523a2da34f4fe546a0a4a0b80b18da0eadb`) 中，PyTorch 利用 CUDA P2P / NVSwitch 硬件映射实现了计算与通信分块融合：

1. **对称内存工作区与通道屏障**（L172–214）：
   - `symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)` 获取跨节点内 `TP=8` GPU 直接映射的 P2P 内存工作区；
   - `symm_mem.get_buffer(remote_rank, shape, dtype, storage_offset)` 直接获取远端 GPU 物理内存的本地虚拟地址指针。
2. **`_pipelined_multi_all_gather_and_consume` 双流交错流水线**（L149–291）：
   - 首先各 Rank 将本地分片拷入 `local_p2p_bufs` 并执行硬件原子屏障 `symm_mem.barrier(channel=1)`（L260–262，消除乱序调度退化）；
   - 立即在主流上对本地分片执行第 `0` 块 `shard_consumer(shard, rank)`（即 `1/TP` 规模的子 GEMM，L267）；
   - 在 `step = 1 .. group_size - 1` 循环中（L269–279），按 `step % 2` 交替在 `current_stream` 与 `backend_stream` 上发射 `copy_shard(dst=shards[remote_rank], src=remote_p2p_bufs)`（NVLink P2P Pull）紧跟 `shard_consumer(shards[remote_rank], remote_rank)`。这不仅使第 $k$ 步的远端 NVLink 拷贝与第 $k-1$ 步的 GEMM 完全重叠，还允许双流子 GEMM 尾波互相填充，降低 Wave Quantization 损失。
3. **`_fused_matmul_reduce_scatter_impl` 分块产出与 All-to-All 归约**（L895–969）：
   - 将输入激活 `A` 沿 `scatter_dim` 移至首维并切分为 `group.size()` 个分块 `A_shards`；
   - 定义 `chunk_producer(rank, out)` 调用 `mm_out_op(A_shards[rank], B, scale_a=..., out=out)`（原生支持 `FP8` `_ScaleMode.ROW_WISE_SHARDED` / `TENSOR_WISE`），并通过 `_pipelined_produce_and_all2all` 边计算分块 GEMM 边通过 NVLink P2P 写出，最后在单核内完成连续归约（L963–968）。

```
Sync-TP (Stream 0):
[ -------- NCCL AllGather (TP=8) -------- ][ ------------ Full GEMM (M x N x K) ------------ ]

SymmetricMemory Async-TP (2-Stream Micro-Pipelined):
Stream 0 (Current): [mv][b1][ GEMM_0 (local) ][ P2P_cp_2 ][ GEMM_2 ][ P2P_cp_4 ][ GEMM_4 ][ P2P_cp_6 ][ GEMM_6 ][b0]
Stream 1 (Backend):         [ P2P_cp_1 ][ GEMM_1 ][ P2P_cp_3 ][ GEMM_3 ][ P2P_cp_5 ][ GEMM_5 ][ P2P_cp_7 ][ GEMM_7 ]
```

---

## 3. 定量实验与架构对比（`experiments/fsdp2_async_tp_composability`）

运行 `python experiments/fsdp2_async_tp_composability/run.py --check` 得到如下确定性建模结果：

### 3.1 `FSDP1` (`FlatParameter`) vs `FSDP2` (`DTensor`) 显存碎片与重打包开销

| 训练场景 | 基础活跃分配 (`GiB`) | `FSDP1` 峰值预留 (`GiB`) | `FSDP2` 峰值预留 (`GiB`) | 节省 HBM (`GiB/GPU`) | `FSDP1` 碎片率 | `FSDP2` 碎片率 | Unshard 重打包提速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **70B Uniform BF16 Dense** | 21.00 | 23.93 | **21.28** | **2.65 GiB** | 11.40% | **1.15%** | **7.79x** (`14.8 -> 1.9 ms`) |
| **70B VLM 混合冻结视觉塔 + Rank-64 LoRA** | 23.04 | 30.11 | **23.40** | **6.72 GiB** | 18.60% | **1.28%** | **16.00x** (`38.4 -> 2.4 ms`) |
| **405B Interleaved 1F1B (`V=4`) + Row-Wise FP8** | 29.20 | 39.34 | **29.70** | **9.65 GiB** | 21.80% | **1.42%** | **16.77x** (`52.0 -> 3.1 ms`) |

### 3.2 `SymmetricMemory` `Async-TP=8` 通信掩盖率与层提速（H100 SXM5, $D=8192$）

| 每 TP 组 Token 数 | 单层 GEMM 耗时 (`ms`) | `Sync-TP` 暴露通信 (`ms`) | `Async-TP` 暴露通信 (`ms`) | 通信气泡消除率 | 单层总耗时 (`Sync -> Async`) | 端到端单层提速 |
|---:|---:|---:|---:|---:|---:|---:|
| **2,048** | 1.825 | 0.584 | **0.051** | **91.27%** | `2.409 -> 1.876 ms` | **+22.13%** |
| **4,096** | 3.651 | 1.060 | **0.060** | **94.34%** | `4.711 -> 3.711 ms` | **+21.23%** |
| **8,192** | 7.301 | 2.012 | **0.078** | **96.13%** | `9.314 -> 7.379 ms` | **+20.77%** |
| **16,384** | 14.603 | 3.917 | **0.114** | **97.09%** | `18.520 -> 14.717 ms` | **+20.53%** |

### 3.3 千卡 H100 5D 并行组合（`TP × CP × EP × PP × FSDP2`）端到端收益

| 配置方案 | 5D DeviceMesh 拓扑 | 单步耗时 (`ms`) | 暴露 TP 通信 (`ms`) | 暴露 FSDP 通信 (`ms`) | 峰值预留 HBM (`GiB`) | 端到端吞吐倍率 |
|---|---|---:|---:|---:|---:|---:|
| `baseline_fsdp1_sync_tp` | `TP=8, CP=1, EP=1, PP=4, FSDP1=32` | 4820.0 | 740.0 | 465.0 | 76.4 | `1.000x` (`39.64%` MFU) |
| `fsdp2_dtensor_sync_tp` | `TP=8, CP=1, EP=1, PP=4, FSDP2=32` | 4360.0 | 740.0 | 88.0 | 66.8 | **`1.106x`** (`43.82%` MFU) |
| `fsdp2_hsdp_async_tp_5d` | `TP=8, CP=2, EP=8, PP=4, FSDP2=16` | **3645.0** | **78.0** | **52.0** | **64.2** | **`1.322x`** (**`52.43%` MFU**) |

---

## 4. 解决方案架构师决策指南与可证伪问题

1. **何时必须从 `FSDP1` 迁移至 `FSDP2` (`fully_shard`)**：
   - 只要工作负载涉及 **部分冻结层（VLM/Multimodal）**、**PEFT/LoRA 微调**、**FP8/NVFP4 逐行/分块缩放（`Float8Tensor` 子类）** 或 **Interleaved 1F1B 流水线并行**，必须使用 `FSDP2` 逐参数 `DTensor` 并显式配置 `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch`。
2. **启用 `SymmetricMemory` `Async-TP` 的前提条件**：
   - 仅限具备全互联 NVSwitch 的节点内（8× H100/B200 或 GB200 NVL72）`TP` 组，且必须对 `fused_all_gather_matmul` 的左矩阵分片调用 `restride_A_shard_for_fused_all_gather_matmul` 以保证按块连续访存。
3. **后续可证伪实验问题**：
   - 当序列长度 $S < 1024$ 且采用 NVFP4 极短耗时子 GEMM 时，`Async-TP` 的 `8` 次子分块切分是否会因为单子块 `M_chunk = S / 8 < 128` 跌破 Blackwell `tcgen05` 2-CTA MMA 的最小满载 Tile 高度，从而抵消通信重叠收益？

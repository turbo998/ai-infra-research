# Megatron-LM MFSDPv2 分阶段 MXFP8 权重平面 All-Gather 与 EP A2A 分段重计算、TransformerEngine 掩码置换容量钳制及 Mooncake/NIXL/LMCache 低延迟通信剖析

> **双语导航 / Bilingual Navigation**: [中文版](megatron-mfsdp-mxfp8-te-permute-mooncake-nixl.md) | [English Version](megatron-mfsdp-mxfp8-te-permute-mooncake-nixl.en.md)
> **证据锚点 / Evidence IDs**: `src-megatron-mfsdp-mxfp8-te-permute-mooncake-nixl-2026` · `ev-megatron-mfsdp-mxfp8-te-permute-mooncake-nixl-2026` (`priority: 73`, Phase 64 Track C)
> **可复现实验 / Reproducible Experiment**: `python experiments/megatron-mfsdp-mxfp8-te-permute-mooncake-nixl/run.py --check`

---

## 1. 架构师问题定义与工程背景

在 Blackwell MXFP8 分布式训练（Megatron-FSDPv2 与 TransformerEngine）、专家并行（EP All-to-All）激活重计算、以及 RDMA/IPC KV Cache 传输与遥测底座（Mooncake, NIXL, LMCache）中，存在五个直接决定跨节点扩展效率与内存安全的架构问题：

1. **MFSDPv2 中 MXFP8 双平面冗余 All-Gather**：MXFP8 权重包含 `rowwise`（前向 $Y = X W^T$ 读取）与 `columnwise`（反向 $dX = dY W$ 读取）两套物理平面（Data + E8M0 Scale）。若每次 `unshard_parameters` 均无差别 All-Gather 全部 4 个平面，将浪费整整一半的跨 GPU 集合通信带宽与非活跃平面显存。
2. **EP All-to-All 通信重叠下的分段全激活重计算（Per-Segment Full Recompute）状态漂移**：在 `--overlap-moe-expert-parallel-comm` 下启用 `--recompute-granularity full` 时，前向层原地修改了 `ModelChunkState` 中的 `input_ids` 与 `mtp_input_mask`（MTP 将两者拼接循环移位以合并 CP 边界交换）。若分段快照漏掉 `mtp_input_mask`，反向重放会再次滚动已移位的掩码，导致 `torch.where` 截断错误的 Embedding 行梯度。
3. **TransformerEngine Triton Mask-Map `te_permute` 忽略 `num_out_tokens` 容量上限引发越界写**：当 MoE 路由开启容量因子（Capacity Factor）使得 `num_out_tokens < routing_map.sum()` 时，Triton `_row_id_map_pass_2_kernel` 此前未对超出 `num_out_tokens` 的目标行号进行钳制，导致后续 Permute 核直接写穿输出张量尾部内存。
4. **Mooncake RDMA `submitPostSend` 锁内逐 WR 读取系统时钟与 Store 元数据分片锁争用**：在持写锁构建 256-WR 发送链时，每个 WR 调用一次 `clock_gettime(CLOCK_REALTIME)`（约 `19 ns/WR`）占据了建链绝大部分耗时并推迟网卡 Doorbell；同时 Store 的 DFS 淘汰与软固定过期若驻留在 1024 个元数据分片锁内会阻塞在线读写。
5. **NIXL `sharedRingBuffer` 读写指针伪共享（False Sharing）与 LMCache 多进程 Lookup 轮询延迟**：SPSC 环形缓冲区的 `write_pos` 与 `read_pos` 紧邻在同一缓存行且每次 `push()`/`pop()` 均执行跨核 `memory_order_acquire` 加载；而 LMCache `LMCacheMPSchedulerAdapter` 此前需等待所有 Server 完成 `LOOKUP` 并推迟到下一个调度器 Step 才发送 `QUERY_PREFETCH_STATUS`。

---

## 2. 核心源码变更与数学建模

### 2.1 Megatron-LM MFSDPv2 分阶段 MXFP8 权重平面 All-Gather (`#7907`, `18bc49d`) 与 EP A2A 分段激活重计算 (`#6429`, `da2c0ec`)

1. **MFSDPv2 按计算阶段（`Phase.FORWARD` / `Phase.BACKWARD`）选择性 All-Gather MXFP8 平面**：
   - 在 `megatron_fsdp/experimental/module.py`、`parameter_group.py` 与 `quantized_dbuffer.py` 中，将单事件 `_unshard_event` 升级为字典 `_unshard_events: dict[Phase, torch.cuda.Event]`；
   - `QuantizedDBuffer.redistribute(..., rowwise=rowwise, columnwise=columnwise)` 仅通信当前阶段所需的物理平面：**前向仅 All-Gather `rowwise_data` + `rowwise_scale_inv`**，**反向仅 All-Gather `columnwise_data` + `columnwise_scale_inv`**；
   - `update_mxfp8_tensor(src, dst, rowwise=..., columnwise=...)` 仅对新收集的平面执行 TE 所需的 Scale Padding，保留另一阶段已安装的平面及其就绪事件，避免覆盖 TE 保存用于反向的 `MXFP8Tensor` 实例。
   - **带宽收益建模**：对于形状 $[14336, 5376]$、块大小 $B=32$ 的 MXFP8 权重，单平面（Data `73.50 MiB` + Scale `2.2969 MiB`）为 **`75.7969 MiB`**，双平面为 **`151.5938 MiB`**。在“前向 Unshard + 反向重 Unshard”的训练 Step 中，单步 All-Gather 通信量从 **`303.1875 MiB`** 降至 **`151.5938 MiB`**，**跨 GPU 集合通信量精确降低 `50.0%`**。
2. **EP A2A 重叠的分段全激活重计算 (`#6429`, `da2c0ec`)**：
   - `megatron/core/models/common/model_chunk_schedule_plan.py` 引入 `RecomputeSegment`，在初始前向使用 `torch.no_grad()` 运行并在退出分段时立即调用 `reset_for_recompute()` 释放激活引用（保留底层 Storage 引用结构以免破坏后续重放图），同时强制对输出执行 `.detach()` 防止内部局部 `enable_grad`（如 Bias-Dropout-Add）残留 `grad_fn` 钉死整段计算图；
   - 快照完整覆盖 5 个原地可变字段 `_MUTABLE_CHUNK_STATE_FIELDS = ("input_ids", "position_ids", "padding_mask", "mtp_input_mask", "mtp_hidden_states")`，消除 MTP 掩码二次滚动引发的静默梯度错位。

### 2.2 TransformerEngine Triton Mask-Map 容量钳制 (`#3347`, `c9c9225`) 与 `GroupedTensor` 视图/流追踪 (`#3701`, `2199b71`)

1. **Triton `_row_id_map_pass_2_kernel` 容量丢弃**：在 `transformer_engine/common/triton/permutation.py` 中传入 `num_out_tokens`，并在写出 `row_id_map` 前执行：
   ```python
   row_id = tl.where(row_id >= num_out_tokens, -1, row_id)
   ```
   与 CUDA `moe_permute_row_map` 保持严格一致（JAX 路径传入 `2**31 - 1` 保持不丢弃）。在 `4096 tokens x 8 experts`（总路由 `16,420`，容量上限 `16,403`）测试中，精确保留前 `16,403` 个目标行并将超出的 **17 个路由项标记为 `-1`**，彻底阻断 17 行越界写。
2. **`GroupedTensor` Autograd 视图与多缓冲区 `record_stream` (`#3701`, `2199b71`)**：`make_wrapper_like` 将初始 `requires_grad` 设为 `False` 交由 PyTorch Autograd 在 `__torch_dispatch__` 返回后附加可微视图元数据，并在 `aten.record_stream` 时遍历全部 **7 个底层张量**（`rowwise_data`, `columnwise_data`, `scale_inv`, `columnwise_scale_inv`, `amax`, `first_dims`, `tensor_offsets`），杜绝多流异步通信下的提前释放重用。

### 2.3 Mooncake RDMA 单链时钟读取 (`#4793`, `dd1217b`) 与无分片锁淘汰 (`#4854`, `c81f4ad`)、NIXL 环形缓冲区优化 (`#2018`, `a9f1f7a`) 及 LMCache 回调式 Lookup (`#5960`, `ff9185d`)

1. **Mooncake `RdmaEndPoint::submitPostSend` 每链单次时钟读取 (`#4793`)**：将 `const int64_t post_ts = getCurrentTimeInNano()` 移出内层 WR 循环（每条 `ibv_post_send` 链仅调用 1 次）。在 256-WR（两条 128-WR 链）基准测试（`author-benchmark`）中，系统调用次数从 `256` 降至 `2`，**持锁与 Doorbell 延迟从 `6.3 us` 降至 `1.9 us`（每 WR 从 `25.0 ns` 降至 `7.5 ns`，提速 `3.3158x`）**。
2. **Mooncake Store 无元数据分片锁淘汰与 `DeferDeletion` (`#4854`)**：通过 `ReadCursor(tenant_id, CursorSlice{index, count})` 并发切片游标将 DFS 淘汰、软固定（Soft Pin）过期与在途任务检查移出 1024 个元数据分片锁，并在 `ImmutableBucketAllocator` 中引入 `DeferDeletion(bucket, not_before)` 延迟物理文件删除直至在途读取租约到期。
3. **NIXL `sharedRingBuffer` 256B 对齐与缓存位置检查 (`#2018`, `a9f1f7a`)**：升级 `TELEMETRY_VERSION = 6`，使用 `alignas(256)` 隔离 `write_pos` 与 `read_pos`（同时将 `version` 固定在字节偏移 `16` 处以保证跨版本头部校验安全），并引入进程本地 `cachedReadPos_` / `cachedWritePos_`——`push()` 仅当 `next_write == cachedReadPos_`（看起来已满）时才执行跨核 `memory_order_acquire` 重载 `read_pos`，在 1024 容量环上连续推送 1000 个事件时将跨核缓存一致性加载减少 **99.9%**（`1000 -> 1`）。
4. **LMCache 回调驱动 `LookupState` (`#5960`, `ff9185d`)**：为 `MessagingFuture` 增加线程安全的 `add_done_callback`；每个 Server 独立运行 `LOOKUP_IN_FLIGHT -> QUERY_PREFETCH_STATUS_IN_FLIGHT -> RESOLVED/TIMED_OUT` 状态机，在传输接收线程收到 `LOOKUP` 确认瞬间立即发出 `QUERY_PREFETCH_STATUS`，无需等待下一个调度器轮询 Step（在 `5.0 ms` Step 周期与 `0.4 ms` RPC 下将查表解析延迟从 `5.40 ms` 降至 `0.80 ms`，提速 **`6.75x`**）。

---

## 3. 确定性实验与验证结果 (`own-offline` + `author-benchmark`)

运行 `python experiments/megatron-mfsdp-mxfp8-te-permute-mooncake-nixl/run.py --check` 输出如下验证摘要：

| 子系统 / 仓库 | 优化机制 | 修复前 / 基线 | 修复后 / 当前实现 | 核心收益 |
|---|---|---|---|---|
| **Megatron-LM MFSDPv2** (`#7907`) | 分阶段 MXFP8 平面 All-Gather | `303.19 MiB/step` | **`151.59 MiB/step`** | **All-Gather 通信量降低 50.0%** |
| **Megatron-LM EP A2A** (`#6429`) | `RecomputeSegment` + 5 字段快照 | 漏快照 `mtp_input_mask` 致二次滚动 | **快照全部 5 个可变字段 + `no_grad` 重放** | 消除 MTP 梯度错位与重放激活泄漏 |
| **TransformerEngine** (`#3347`, `#3701`) | Triton Mask-Map 容量钳制 + 流追踪 | 超容量 17 行越界写；包装器漏流追踪 | **`row_id >= num_out_tokens -> -1`；追踪 7 缓冲区** | 阻断 Permute OOB 与多流异步 UAF |
| **Mooncake RDMA** (`#4793`) | 每条 `ibv_post_send` 链读 1 次时钟 | `6.3 us` (`25.0 ns/WR`, 256 次调用) | **`1.9 us` (`7.5 ns/WR`, 2 次调用)** | **持锁建链与 Doorbell 提速 `3.32x`** |
| **NIXL & LMCache** (`#2018`, `#5960`) | `alignas(256)` 缓存游标 + Future 回调 | 每次 Push 跨核加载；等下一调度 Step | **跨核加载 `-99.9%`；回调延迟 `5.4 -> 0.8 ms`** | 消除遥测伪共享与跨 Step 查表泡沫 |

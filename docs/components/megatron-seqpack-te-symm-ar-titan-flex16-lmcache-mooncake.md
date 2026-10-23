# Megatron-LM 序列打包 PP 中间级守卫、TransformerEngine 1/4 步长对称 AllReduce 缓存分桶与 GroupedTensor 微分视图、TorchTitan FlexAttention 16 图全图捕获与 LMCache/Mooncake/NIXL/Dynamo 深度解析

> **Phase 61 · Priority 70 · Track C（Distributed Training / KV Cache / Collective Communication）**
> **双语版本**: [中文正文](./megatron-seqpack-te-symm-ar-titan-flex16-lmcache-mooncake.md) | [English Version](./megatron-seqpack-te-symm-ar-titan-flex16-lmcache-mooncake.en.md)
> **可复现实验**: `python experiments/megatron-seqpack-te-symm-ar-titan-flex16-lmcache-mooncake/run.py --check`

---

## 1. 架构背景与核心问题

在大规模流水线并行（PP）序列打包训练、Blackwell/NVLink 对称内存通信、PyTorch `torch.compile` + CUDA Graph 捕获以及分布式 KV 缓存（LMCache、Mooncake、NIXL、Dynamo）生产部署中，系统栈面临五项关键架构问题：

1. **Megatron-LM 序列打包调度器在 `PP > 2` 中间流水级崩溃**：当启用 `sequence_packing_scheduler` 时，`is_dataset_built_on_rank()` 此前因 `return (... or args.sequence_packing_scheduler is not None) and mpu.get_tensor_model_parallel_rank() == 0` 在所有 PP 阶段（包括中间级 $1 \dots \text{PP}-2$）构造数据集迭代器，不仅浪费主机内存/I/O，而且与仅在首尾阶段期望数据迭代器的流水线调度器发生冲突。
2. **TransformerEngine `symmetric_all_reduce` 动态尺寸缓存膨胀与 `GroupedTensor` Autograd 视图断裂**：在 MoE 等动态 token 数训练中，按精确 `tensor.numel()` 缓存对称内存缓冲区会为每个出现过的行数永久保留一份 `torch.ops.symm_mem.empty` 张量（在 `1..32` 行下累计达最大张量的 `16.5x`）；同时对 `GroupedTensor` 执行切片/转置（如 `t[:]`）时，由于在 `_make_wrapper_subclass` 内部过早设置 `requires_grad`，PyTorch Autograd 跳过挂接 `CreationMeta` 微分视图元数据，导致 `view.is_leaf == True` 且反向梯度丢失。
3. **TorchTitan Muse Spark 在 GB300 上的 FlexAttention 重编译回退与 CUDA Graph 捕获失败**：Muse Spark 在单进程训练+评测+自回归生成中混合使用 2 种注意力掩码（`sliding_window_2048` 与 `global`）× 5 种执行模式（训练、评测、Prefill、Decode `T=8`、Decode `T=256`），共需编译 10 个不同的 `FlexInnerAttention` 图；原 `_COMPILE_RECOMPILE_LIMIT = 8` 导致第 9、10 个图退回 eager 模式并使 GB300 Decode CUDA Graph 捕获报错。
4. **LMCache `/cache/delete` 越过淘汰控制器与 Mooncake 跨 TP/PP/Host 异构 KV 重分片缺失**：LMCache `/cache/delete` 此前直接挂在 `controller_manager` 上，绕过了 `FleetEvictionController` 的 Pin 表保护；Mooncake Store 在 Prefill 与 Decode 采用不同 `TP/PP` 或不同 Host 内存布局（`PLHD`/`LPHD`/`HPLD`）时缺少直接基于存储副本的切片重分片规划器与租户控制面绑定校验。
5. **NIXL UCX 同步/重发完成事件漏标与 Dynamo Token 帧 12 个冗余 `null` 字段开销**：NIXL `tracing_nsys` 在 50 次 UCX 传输中漏记了 17 次同步完成/重发/释放路径的 `nixl::xfer.complete` 事件，且每个 UCX 引擎用 `pipe(2)` 占用 2 个 FD；Dynamo Worker 每生成一个 Token 都序列化 12 个 `None` 可选字段（251 字节/帧），且 `ReplicaIngress` 未过滤自身发布的 NATS 事件。

---

## 2. 上游源码级实现与公式推导

### 2.1 Megatron-LM 序列打包 PP 首尾级守卫与 Nemotron MIMO RoPE 透传（`ede2c586cf70`, `964b1deefb95`）

1. **序列打包 PP 阶段守卫（`megatron/training/training.py`）**：
   - 将 `is_dataset_built_on_rank(vp_stage=None)` 收敛为：当 `args.sequence_packing_scheduler is not None` 时，仅在 `is_first_or_last_pipeline_stage(vp_stage) and mpu.get_tensor_model_parallel_rank() == 0` 的 Rank 上构建数据集（在 `PP=4` 下活跃迭代器阶段从 `[0, 1, 2, 3]` 降至 `[0, 3]`，减少 50% 冗余数据集构建并消除中间级调度器异常）。
2. **Nemotron MIMO VLM RoPE 参数透传（`964b1deefb95`）**：在 `get_language_model_config` 中将 CLI 的 `position_embedding_type`、`rotary_base`、`rotary_percent` 与 `seq_len_interpolation_factor` 显式注入语言模型 `TransformerConfig`。

### 2.2 TransformerEngine 1/4 步长对称内存分桶与 `GroupedTensor` 微分视图（`fd7d03b1fa8e`, `2199b7166417`, `f0b1096128c0`）

1. **`symmetric_all_reduce` 四分之二次幂步长分桶（`_SymmMemAllReduce.get_symm_tensor`）**：
   - 给定元素总数 $N = \max(1, \text{tensor.numel()})$，步长与分桶容量定义为：
     $$\text{step} = 2^{\max(0, \lfloor \log_2 N \rfloor - 2)}, \quad C(N) = \left\lceil \frac{N}{\text{step}} \right\rceil \cdot \text{step}$$
   - 在每个 2 的幂次区间 $[2^k, 2^{k+1})$ 内最多生成 **4 个桶**，单缓冲区最大对齐填充开销严格受限于 $\le 25\%$（在 `rows ∈ [1..32], width=256` 下实测最大填充率仅 **17.65%**）。
   - 对于 `1..32` 行动态输入，缓存槽位数从 `32` 降至 `16`，总保留对称内存元素数从 `135,168`（最大张量的 `16.5x`）降至 `49,152`（最大张量的 **`6.0x`**，**削减 63.64% 对称内存驻留**）。
2. **`GroupedTensor` 微分视图与多流安全（`2199b7166417`）**：
   - 在 `GroupedTensor.make_wrapper_like` 中传 `requires_grad=False` 给 `torch.Tensor._make_wrapper_subclass`，使 `__torch_dispatch__` 返回后由 PyTorch C++ Autograd 引擎正确附加 `DifferentiableViewMeta`（`view._base is t` 且 `view.grad_fn is not None`）；同时重写 `record_stream(stream)` 遍历全部底层缓冲张量（`quantized_tensors`、`rowwise_data`、`columnwise_data`、`scale_inv` 等）。
3. **`torch.compile` NVFP4 3D/4D `Linear` 元数据展开（`f0b1096128c0`）**：修复 compiled 路径下 3D/4D 输入的 2D 展平形状、非训练模式 `update_workspace=False` 透传及权重块缩放因子 `swizzled_scales` 保持。

### 2.3 TorchTitan FlexAttention 16 图全图捕获与 Cuspy CUPTI 图注解（`aabe6eff1e6e`, `b93d5c0cf518`）

- 将 `FlexInnerAttention` 的 `_COMPILE_RECOMPILE_LIMIT` 从 `8` 提升至 **`16`**，并在默认与确定性模式下统一启用 `torch.compile(..., fullgraph=True, options={"recompile_limit": 16})`。Muse Spark 所需的 `2 × 5 = 10` 个注意力图全部完成编译而不再触发第 9、10 图的 eager 回退，使 GB300 Decode `T=8` 成功捕获进 CUDA Graph（单步延迟 **`30.9 μs`**）。
- 新增 `capture_graph(..., annotation_config)`，支持在 CUDA Graph 捕获期间通过 CUPTI `correlation_id` 精确映射 `sourceGraphNodeId` 与执行图 `exec_graph_id`。

### 2.4 LMCache 淘汰路由归属、Mooncake 跨 TP/PP 重分片、NIXL 完整追踪与 Dynamo 紧凑帧

1. **LMCache `/cache/delete` 移交 `FleetEvictionController`（`d05401790fad`）**：将 `/cache/delete` 从 `controller_manager` 迁入 `eviction_http_api.py`，在删除前强制检查 Pin 表保护；当未构建淘汰控制器时不再暴露裸删除端点。
2. **Mooncake Store 跨 TP/PP/Host 布局 KV 重分片与租户控制面强制绑定（`e2d6e8e0f2df`, `edd323921d53`）**：
   - 新增 `plan_store_kv_reshard()`，支持源/目标任意 `TP × PP` 网格及 `PLHD`、`LPHD`、`HPLD` 三种主机张量布局的字节级拷贝规划（带 `KVCacheStorePlanningLimits` 上限：`max_parts=100,000`、`max_fragments=100,000`、`max_bytes=1 TiB`）。在单页 `16` tokens、`TP2 → TP4`（每源分片 4 头 → 每目标 Rank 2 头，`head_bytes = 256 B`）下，自动生成步长 `1024 B`、单 token 连续切片 `512 B`、每层每分量 `8,192 B` 的零冗余传输计划。
   - 在 `mooncake-store-rs` 中新增 `ensure_control_plane_tenant()`，拦截非绑定租户对共享控制面 RPC 的越权覆写。
3. **NIXL `nixl::xfer.complete` 全路径标记与 `eventfd` 瘦身（`f8248873bd5b`, `8a78b338a367`）**：
   - 在 `postXferReq` 同步立即完成、`repost` 重发完成与 `releaseXferReq` 释放已完成句柄时均补齐 `mark_xfer_complete_once()`，将 50 次 UCX 请求的完成追踪率从 `33/50`（66.0%）提升至 **`50/50`（100.0%）**；并用单 FD `eventfd(0, EFD_NONBLOCK)` 替代双 FD `pipe(2)`（FD 占用减半 `2 → 1`）。
4. **Dynamo Worker 紧凑 Token 帧与 `ReplicaIngress` 自发事件过滤（`3c7736d89e2b`, `ce67fe87df38`）**：
   - 为 `LLMEngineOutput` 的 12 个可选字段添加 `#[serde(default, skip_serializing_if = "...")]`，使普通解码 Token 帧从 `251 B` 降至 **`20 B`（缩减 92.03% 序列化负载）**；并在 `ReplicaIngress` 中过滤 `envelope.publisher_id == local_publisher_id` 的自发事件。

---

## 3. 确定性实验结果与对比矩阵

运行 `python experiments/megatron-seqpack-te-symm-ar-titan-flex16-lmcache-mooncake/run.py --check` 得到的验证数据如下：

| 评估维度 | 修复 / 优化前基线 | Phase 61 修复 / 优化后 | 关键量化收益 / 架构不变式 |
| :--- | :--- | :--- | :--- |
| **Megatron-LM 序列打包数据集构建 (`PP=4`)** | 全部 4 个 PP 阶段 `[0,1,2,3]` 构建迭代器 | **仅首尾阶段 `[0,3]` 构建数据集** | **消除中间 PP 阶段调度器崩溃**，减少 50% 冗余迭代器 |
| **TE `symmetric_all_reduce` 动态行数缓存 (`1..32` 行)** | 32 个精确桶，保留 `135,168` 元素 (`16.5x`) | **16 个 1/4 步长桶，保留 `49,152` 元素 (`6.0x`)** | **对称内存驻留降低 63.64%**，单桶最大填充率 `17.65% <= 25%` |
| **TorchTitan Muse Spark FlexAttention 图编译** | `recompile_limit=8`（2 个 Decode 图退回 eager） | **`recompile_limit=16, fullgraph=True`（10/10 全图）** | **0 eager 回退**，GB300 Decode `T=8` 捕获仅 **`30.9 μs`** |
| **NIXL UCX 传输完成事件追踪 (`50` 次传输)** | 仅记录 `33/50` (`66.0%`)，漏记 17 次同步/释放完成 | **`mark_xfer_complete_once` 记录 `50/50` (`100%`)** | **100% 传输完成可观测性** + `eventfd` 唤醒 FD 减半 (`2→1`) |
| **Dynamo Worker 单 Token 线协议帧大小** | `251 B`（携带 12 个 `null` 字段与默认枚举） | **`20 B` (`{"token_ids":[1234]}`)** | **每 Token 网络 JSON 载荷降低 92.03%** |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **计算练习**：当 `symmetric_all_reduce` 收到 `numel = 5,000` 个元素的张量时，按 TransformerEngine 的 1/4 步长分桶算法，其 `step`、分配容量 `capacity` 以及对齐填充百分比分别是多少？
   - *解答*：`5000.bit_length() = 13`，$\text{step} = 2^{13 - 3} = 1024$，$C(5000) = \lceil 5000 / 1024 \rceil \times 1024 = 5 \times 1024 = 5120$。填充开销为 $(5120 - 5000) / 5000 = 2.4\% \le 25\%$。
2. **可证伪假设**：在自定义 `torch.Tensor` 子类的 `__torch_dispatch__` 实现中，若对 `aten.slice.Tensor` 等视图算子返回的新包装子类直接在 `_make_wrapper_subclass` 中传入 `requires_grad=True`，能否保留原张量的反向传播图？
   - *证伪验证*：不能。在 dispatch 内部直接把输出标记为 `requires_grad=True` 会使外部 Autograd 包装器认为该张量已处理完毕，从而跳过绑定 `DifferentiableViewMeta`，导致新视图变成无 `grad_fn` 的叶子节点并截断梯度流。

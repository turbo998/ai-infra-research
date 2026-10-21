# Megatron-LM 混合 DSA-GQA 与 GDN 预填充 CUDA Graph、Mooncake Store 跨拓扑 KV 重分片与租户配额及 NIXL 传输遥测剖析

> **关联上游提交（2026-10-20 ~ 2026-10-21）**：
> - `NVIDIA/Megatron-LM` PR `#7156` (`d14b699755061d55cb918f68500fba24401a1846`): `feat: Support DSA over GQA in hybrid models and fix DSA indexer loss logging (#7156)`
> - `NVIDIA/Megatron-LM` PR `#8353` (`29e1a7b6697c61d184c95726865d417fc51b8574`): `feat(ssm): support CUDA graph replay for GDN dynamic inference prefill via single-update GDP (#8353)`
> - `kvcache-ai/Mooncake` PR `#4762` (`e2d6e8e0f2df113d84f9e39379f98c4b67b199ab`): `feat(reshard): add Store KV cache reuse geometry planner (#4762)`
> - `kvcache-ai/Mooncake` PR `#4809` (`44c4d9cf4ace3ce2955312a2467b1305d910dd48`): `feat(store): integrate TenantQuotaPolicy with upsert carry-over and watermark eviction (#4809)`
> - `ai-dynamo/nixl` PR `#2479` (`03d9b74648cc06daef8aa71f15a53c5a98be1319`): `feat(telemetry): add nixl.xfer_request_id and nixl.xfer_execution counters (#2479)`

---

## 1. 核心架构问题与背景

在混合架构（Gated DeltaNet + DSA-GQA）训练/推理以及分布式 KV Cache 存储池（Mooncake Store + NIXL）中，系统层面临三项核心协同难题：

1. **混合模型 DSA-GQA 适配缺失与 GDN 变长预填充无法重放 CUDA Graph**：
   - DeepSeek Sparse Attention (DSA) 此前仅绑定 MLA（Multi-Latent Attention），无法直接嵌入使用 GQA 的 Mamba/GDN 混合架构；同时 `DSAIndexerLossLoggingHelper` 直接对乘过 `loss_scale` 的张量做日志归约，导致监控指标被缩放系数扭曲；
   - Megatron-LM 的 Gated DeltaNet (`gdn.py`) 动态推理预填充原本调用 `causal_conv1d` 与 `chunk_gated_delta_rule`，其工作区分配与启动网格依赖每次调用的序列长度，无法在固定容量的 CUDA Graph 中安全捕获与重放。
2. **Mooncake Store 跨 TP/PP 拓扑 KV 复用冗余写入与多租户 Upsert 覆写配额逃逸**：
   - 当源集群存在 KV 头副本（如 GQA 复制或 TP 冗余，`replica_ordinal > 0`）时，若所有 Rank 均向 Store 上传会产生倍数级冗余写入；而目标集群在不同 TP/PP 切分下加载时，若采用笛卡尔积扫描匹配片段会产生高昂规划开销；
   - 在多租户共享内存池中，对同一 Key 执行 Upsert 覆写时，在新副本尚处于 `PROCESSING` 写入状态期间，旧对象占用的内存尚未真正释放；若此时过早扣减旧配额，租户可通过并发 Upsert 瞬时突破物理内存水位。
3. **NIXL 多次重投递（Repost）传输的遥测因果追踪缺失**：NIXL 的单个 `nixlXferReqH` 请求句柄可通过 `postXferReq` 反复重投递，缺乏稳定非零的 64 位请求 ID 与每次实际后端执行的单调递增序号，导致慢传输根因分析无法区分“投递前校验失败”与“第 $k$ 次重投递硬件完成”。

---

## 2. 数学模型与核心设计机制

```
+--------------------------------------------------------------------------------------------------+
|  Megatron DSA-GQA + GDN Graph (#7156/#8353) | Mooncake Reshard + Quota (#4762/#4809) | NIXL      |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [1] Megatron-LM Hybrid DSA-GQA & GDN Single-Update GDP CUDA-Graph Prefill                       |
|      * DSGroupedSelfAttention + SimplifiedDSGQAIndexer + raw_values unscaled loss logging        |
|      * GDN = GDP with num_householder = 1: preallocated metadata.gdp_chunk_indices / offsets     |
|      * causal_conv1d_varlen_fn + chunk_gated_delta_product_varlen write state in place           |
|                                                                                                  |
|  [2] Mooncake Store KV Reshard Planner (#4762) & TenantQuotaPolicy (#4809)                       |
|      * Upload: filter `part.replica_ordinal == 0` -> 1 writer per unique shard                   |
|      * Load: `replicas=True` + `HeadIntervalIndex.cover(head_start, end)` -> fill all replicas   |
|      * Quota words: Held = policy_words[0], Carried = policy_words[1] (kept while PROCESSING)    |
|      * Watermark eviction: evict (used_ratio - (watermark - eviction_ratio)) * effective_quota   |
|                                                                                                  |
|  [3] NIXL Transfer Telemetry (#2479)                                                             |
|      * 64-bit `nixl.xfer_request_id` (redraws if PRNG yields 0) + atomic `nixl.xfer_execution`   |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 Megatron-LM 混合 DSA-GQA 与 GDN 单次 Householder GDP 图安全预填充（PR `#7156` / PR `#8353`）

- **DSA-GQA 混合注意力与无偏损失监控（PR `#7156`）**：新增 `DSGroupedSelfAttention`、`DSGQACoreAttention` 与 `SimplifiedDSGQAIndexer`，支持在 GQA 层上直接挂载 Top-$k$ 稀疏索引器，并定义未融合内核后端集合 `_UNFUSED_DSA_KERNEL_BACKENDS = ("none", "min-memory-triton", "min-memory-torch", "reference")`；在 `DSAIndexerLossLoggingHelper` 中新增 `raw_values` 字典保存未乘 `loss_scale` 的原始辅助损失，消除日志失真。
- **GDN 预填充转化为 `num_householder = 1` 的 GDP 图重放（PR `#8353`）**：由于 Gated DeltaNet 在数学上等价于单次 Householder 更新（`num_householder = 1`）的 Gated DeltaProduct，`gdn.py` 将 `ssm_prefill` 切换至 `causal_conv1d_varlen_fn`（使用预计算的 `conv_seq_idx` / `conv_seq_start`）与 `chunk_gated_delta_product_varlen`。在图重放前，推理上下文原地刷新固定容量的块描述符（`gdp_chunk_indices`, `gdp_chunk_indices_dp`, `gdp_chunk_offsets`），并在主机端以 FP32 预计算门控：

$$g = -\exp(A_{\log}) \cdot \operatorname{softplus}(\alpha + \text{dt\_bias}), \quad \beta = \sigma(\beta_{\text{raw}})$$

同时在存在 GVA 头扩展时对 $Q, K$ 执行 `repeat_interleave(head_repeats, dim=2)`，由内核直接根据 `state_indices = batch_indices` 原地更新 `ssm_state`，彻底消除了动态形状工作区分配。

### 2.2 Mooncake Store 跨拓扑 KV 重分片规划器与租户配额策略（PR `#4762` / PR `#4809`）

- **Store 上传去重与区间索引加载规划（PR `#4762`）**：
  - `plan_kv_cache_store_upload` 调用 `_selected_parts(..., replicas=False)`，仅选取 `part.replica_ordinal == 0` 的唯一头分片写入 Store，避免冗余副本重复上传；
  - `plan_kv_cache_store_load` 调用 `_selected_parts(..., replicas=True)`，按 `(global_layer_id, KVCacheComponent)` 构建排序区间树 `HeadIntervalIndex`，对每个目标副本的 `[head_start, head_start + head_count)` 执行精确区间覆盖查询，并按相对字节偏移：
    $$\text{part\_offset}' = \text{fragment.part\_offset} + (\text{start} - \text{fragment.head\_start}) \cdot \text{fragment.head\_stride\_bytes}$$
    生成零冗余加载计划，同时强制校验 `max_parts = 100,000`、`max_fragments = 100,000` 与 `max_bytes = 1 TiB` 契约上限。
- **双字租户配额状态机与高低水位驱逐（PR `#4809`）**：
  每个 Key 的 `owner_state().policy_words` 维护两个 64 位计费字：`Held = policy_words[0]`（当前账户为该 Key 持有的配额）与 `Carried = policy_words[1]`（Upsert 覆写时被替换旧对象尚未结清的欠款）。当 `OnReplace` 发生时，$\text{Carried} \mathrel{+}= \operatorname{MemoryCharge}(\text{old\_meta})$；在 `OnWriteRelease` 中：
  $$\text{Carried} \leftarrow \begin{cases} \text{Carried}, & \text{若 }\operatorname{HasProcessingMemoryReplica}(\text{new\_meta}) \\ 0, & \text{否则} \end{cases}, \quad \text{owed} = \operatorname{SaturatingAdd}(\operatorname{MemoryCharge}(\text{new\_meta}), \text{Carried})$$
  若 $\text{owed} > \text{Held}$，调用 `account.ChargeUnchecked(owed - Held)` 强制入账而不关闭准入开关；若 $\text{owed} < \text{Held}$，立即调用 `account.Release(Held - owed)` 释放差额。后台 `OnMaintenance` 发现租户使用率 $\text{used\_ratio} > \text{watermark}$ 时，一次性驱逐至目标水位 $\text{target\_ratio} = \max(0, \text{watermark} - \text{eviction\_ratio})$：
  $$\text{target\_bytes} = \big(\text{used\_ratio} - (\text{watermark} - \text{eviction\_ratio})\big) \cdot \text{effective\_quota\_bytes}$$

### 2.3 NIXL 64 位非零请求 ID 与单调执行计数器（PR `#2479`）

NIXL 在创建传输请求时生成 64 位随机 `nixl.xfer_request_id`（若伪随机数生成器抽中 `0` 则循环重抽，保留 `0` 作为“未分配”哨兵值），并维护原子计数器 `nixl.xfer_execution`（仅在通过重投递与通知参数校验、即将调用 `backend->postXfer` 前自增 `1`），将其同步写入 `nixl::xfer.complete` 遥测事件。

---

## 3. 确定性实验与量化基准验证

运行 `python experiments\megatron_dsa_gqa_gdn_mooncake_store_reshard_quota\run.py --check` 验证全部核心不变量：

| 评估模块 | 验证场景 | 基线 / 缺陷行为 | 修复 / 规划后行为 | 量化验证结论 |
|---|---|---|---|---|
| **Megatron GDN 图重放 (`#8353`)** | 3 条变长序列 `cu_seqlens=[0,110,174,310]`, `chunk_size=64`, 容量 `12` | 动态块数导致每次重捕 CUDA Graph | **`6` 个有效块 + `6` 个 `(-1,-1)` 填充槽位，`num_householder=1`** | **100% 固定网格与工作区容量** |
| **Megatron DSA 损失日志 (`#7156`)** | 4 层原始损失 `[0.42,0.38,0.35,0.45]`, `loss_scale=0.1` | 误记录缩放均值 `0.04`（失真 `10x`） | **`raw_values` 还原真实均值 `0.40`** | **消除 `loss_scale` 对监控指标的耦合** |
| **Mooncake Store 重分片 (`#4762`)** | 源 `TP=2`（含 2 副本共 4 分片）$\to$ 目标 `TP=4`（含 2 副本共 8 分片） | 源端 4 分片重复上传 (`2x` 写放大) | **上传去重为 `2` 个 Writer，加载生成 `32` 条精确区间 Range** | **上传流量减少 `50.0%`，目标副本 100% 覆盖** |
| **Mooncake 租户配额 (`#4809`)** | 配额 `1000B`，旧对象 `400B` 被 `900B` Upsert 覆写（水位 `0.85`, 驱逐比 `0.15`） | 写入期间漏计旧对象或按单对象赤字抖动驱逐 | **`PROCESSING` 期计费 `1300B`，落盘后结清为 `900B`，驱逐 `200B` 至 `700B`** | **彻底杜绝并发 Upsert 内存透支与驱逐抖动** |
| **NIXL 传输遥测 (`#2479`)** | 连续 2 次抽中 `0` + 4 次调用（含 2 次前置校验失败） | `request_id=0` 混淆未初始化，失败调用污染计数 | **重抽 `2` 次得非零 ID，`xfer_execution` 精确计为 `2`** | **请求级与执行级遥测 100% 精确对齐** |

---

## 4. 生产落地指南与适用边界

- **适用场景**：Megatron-LM 混合 GDN + DSA-GQA 模型训练与动态推理 CUDA Graph 加速、Mooncake Store 跨异构 TP/PP 集群的 Prefix KV 缓存共享与多租户内存隔离，以及 Dynamo/NIXL 跨节点 RDMA 重投递链路追踪。
- **不适用 / 边界条件**：Megatron GDN 的 `chunk_gated_delta_product_varlen` 图重放要求当前批次总块数不超过预分配的静态块描述符容量，且动态推理下不支持 Chunked Prefill（`assert not context.is_chunked_prefill_enabled()`）。

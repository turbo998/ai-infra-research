# Megatron-LM 原位词表并行交叉熵与 MTP 早反向传播、Kimi-K3 全局 Batch 分位数均衡直方图、CUDA Graph MoE GTP 拆分预取链、LMCache 异常安全引用计数与 Mooncake 切片租户游标深度解析

- **研究日期**：2026-10-24（Phase 58 · Priority 67 · Track C）
- **上游证据链**：
  - `NVIDIA/Megatron-LM@6e5946498f` (`#7478`, *perf(ce,mtp): keep VocabParallelCrossEntropy in-place under torch.compile and add mtp_loss_early_backward*)
  - `NVIDIA/Megatron-LM@92641a421f` (`#7394`, *feat(moe): Kimi K3 global-batch quantile balancing and CUDA graph capture cleanup*)
  - `NVIDIA/Megatron-LM@9e5aad0c6a` (`#8345`, *fix(gtp): split graphed MoE expert FC1/FC2 prefetch chains*)
  - `NVIDIA/Megatron-LM@85bf0a667c` (`#8344`, *refactor(ssm): share Mamba branch execution via local ResidualBranchInput*)
  - `NVIDIA/Megatron-LM@3630e44a80` (`#7736`, *fix(dist): forward timeout to DYNAMIC_DP_CP_GROUP*)
  - `LMCache/LMCache@562517ad9f` (`#5940`, *fix(storage): release memory_objs in batched_put try/finally*)
  - `LMCache/LMCache@48fc9d4849` (`#5696`, *fix(vllm): support participates_in_prefix_caching fallback in is_scratch_spec*)
  - `kvcache-ai/Mooncake@c81f4ad486` (`#4854`, *refactor(master): move DFS eviction and soft-pin off metadata shards onto sliced ReadCursor and DeferDeletion*)
  - `kvcache-ai/Mooncake@e764353f4e` (`#4858`, *feat(master): add HTTP /api/v1/soft_pin and /api/v1/soft_unpin and raise max TTL to 7d*)
  - `kvcache-ai/Mooncake@5872ee870d` (`#4844`, *fix(store): bound processing replica cleanup via tenant iteration*)
- **证据等级**：`implementation` + `own-offline`（`experiments/megatron-qb-histogram-mtp-early-bwd-gtp-chains-lmcache-mooncake-dfs/run.py` 确定性验证）

---

## 1. 架构背景与核心问题

在大规模 MoE 预训练与分布式 KV 存储平面中，Phase 58 Track C 攻克了五个关键系统瓶颈：

1. **`torch.compile` (`Inductor`) 下词表并行交叉熵的 `4 GiB` 额外瞬态分配与 MTP 多头 `[S, B, V]` 激活堆积（`Megatron-LM #7478`）**：
   - 当被编译的单个计算图同时对 `[S, B, V]` logits 执行原位修改（`vocab_parallel_logits -= logits_max; exp_()`）以及行归约（`exp_logits.sum`）或目标散射（`grad[idx] -= ...`）时，PyTorch Inductor 拒绝保持原位语义，而是分配一块全新的 `[S, B, V]` 缓冲区再拷回输入——对于 `8K x 128K` FP32 词表头，这意味着额外消耗 **`4.0 GiB`** 瞬态显存并退化约 **`2x`** 耗时。
   - 同时，开启 Multi-Token Prediction (`mtp_num_layers = K`) 时，默认延迟反向传播必须把主头和全部 `K` 个 MTP 头的 `[S, B, V]` FP32 softmax 张量一直保活到主反向传播开始，峰值显存随 `K` 线性增加 `K * 4.0 GiB`。
2. **Kimi-K3 全局 Batch 分位数路由均衡在 Dropless 重试与失败 CUDA Graph 捕获下的状态污染（`Megatron-LM #7394`）**：全局 Batch 分位数均衡需要在所有微批次上累积 `[num_experts, 1000]` 的 `int32` 直方图；若 dropless MoE 发生缓冲区溢出触发全迭代重跑（`prepare_for_rerun()`），未清零的直方图会导致失败尝试的 token 被重复计数。此外，`create_cudagraphs()` 若在捕获中途抛异常，未关闭的捕获标志会导致后续常规反向传播的 DDP 梯度归约 hook 永久失效。
3. **Generalized Tensor Parallelism (GTP) 在 CUDA Graph 捕获下将 MoE FC1/FC2 混入同一链导致死锁/错位（`Megatron-LM #8345`）**：此前被捕获的 MoE 专家参数全部归入单一 `GTP_graphed` 链，破坏了 grouped-expert 预取的一步超前（one-block-ahead）拓扑契约。
4. **LMCache `StorageManager.batched_put` 后端抛异常导致固定内存块永久泄漏（`LMCache #5940`）**：`batched_put` 在调用各存储后端前将所有 `memory_objs` 的引用计数加一，但若某个后端抛出异常，后续的 `ref_count_down()` 被跳过，造成整批 pinned memory 块无法回收。
5. **Mooncake Master 元数据分片扫描锁争用与 DFS 桶淘汰后的并发读取竞态（`Mooncake #4854, #4858`）**：直接遍历 `kNumShards = 1024` 个元数据分片处理 DFS 淘汰与 soft-pin 过期会阻塞高并发读写，且立即物理删除被淘汰的 DFS 桶文件会导致仍在读取副本的客户端报错。

---

## 2. 核心实现机制与源码走读

### 2.1 Megatron-LM 词表并行交叉熵原位拆分与 `mtp_loss_early_backward`（`#7478`）

1. **三段式 `jit_fuser` 原位拆分（`megatron/core/fusions/fused_cross_entropy.py`）**：
   - 将 `VocabParallelCrossEntropy` 拆分为三个独立的编译算子：只读的 `_gather_predicted_logits`、纯逐元素的 `_exp_logits_inplace` 与纯逐元素的 `_scale_gradients_inplace`，而将 `gather` / `scatter` / `row-sum` 放在编译图外部执行。
   - 由于被编译子图内只含逐元素操作，Inductor 严格保持原位读写，彻底消除 `S*B = 8192, V_local = 131072` FP32 下的 **`4.0 GiB`** 瞬态缓冲区（峰值从 `8.0 GiB` 降至 `4.0 GiB`）。
2. **`mtp_loss_early_backward=True` 与 `_MTPHeadGradInjector`（`multi_token_prediction.py`）**：
   - 在前向 `process_mtp_loss` 循环中，对每个 MTP 深度 $k \in \{1 \dots K\}$ 立即以脱离图（detached）的 `head_hidden` 执行词表投影 + 交叉熵 + `backward()`：
     - 当前 MTP 头的 `[S, B, V]` logits 与 softmax 在进入下一个头 $k+1$ 之前立即释放；
     - 共享输出层权重的梯度当场累加；
     - 仅保留 `[S, B, H]` BF16 的输入梯度 `head_grad`（`8192 * 4096 * 2 B = 64 MiB`），并通过自定义 `_MTPHeadGradInjector.apply(output, head_hidden, head_grad)` 注入主反向传播图。
   - 对于 `K = 3` 个 MTP 头，词表头相关的常驻激活峰值从 `(1 + 3) * 4.0 GiB = 16.0 GiB` 骤降至 `4.0 GiB + 3 * 0.0625 GiB = 4.1875 GiB`（**降低 `73.83%`**），且总计算 FLOPs 完全不变（零重计算）。

### 2.2 Kimi-K3 全局 Batch 分位数均衡直方图与 GTP 拆分捕获链（`#7394`, `#8345`）

1. **Kimi-K3 全局 Batch 直方图池化（`#7394`）**：
   - 当 `--moe-router-load-balancing-type quantile_balancing --moe-router-quantile-balancing-estimation-scope global_batch` 启用时，每个 MoE 路由器注册 `qb_histogram`（形状 `[num_experts, moe_router_qb_num_bins=1000]`，`int32`，`256 * 1000 * 4 B = 1,000 KiB`/层）。
   - 在 `finalize_model_grads.py` 的 `_update_router_expert_bias_with_quantile` 中跨 DP/CP/EP 通信组一次性池化所有本地路由器的直方图并按分位数 $q = 1 - \text{topk} / E$ 更新 `expert_bias`。
   - 在 `paged_stash.py` 的 `prepare_for_rerun()` 中调用 `_reset_qb_histograms()` 将直方图原地清零（保留 CUDA 指针地址以兼容 CUDA Graph），并在 `cuda_graphs.py` 的 `create_cudagraphs()` 外层包裹 `try...finally` 块，确保捕获失败时强制调用 `_set_capture_end()` 与 `te_set_capture_end()`。
2. **拆分 Graphed MoE 预取链（`#8345`）**：
   - `generalized_tensor_parallelism.py` 的 `get_graphed_chain_ids()` 将捕获链扩展为三元组：`("GTP_graphed", "GTP_remat_grouped_fc1_graphed", "GTP_remat_grouped_fc2_graphed")`，其中 FC1 与 FC2 各自拥有独立的双链表（避免跨层拓扑串扰），但通过 `_stream_key(chain_id, group)` 共享同一条 IB 通信流。

### 2.3 LMCache 批量写入 `try...finally` 护栏（`#5940`）与 Mooncake 切片游标 + 延迟删桶（`#4854, #4858`）

1. **LMCache `StorageManager.batched_put` 异常安全引用计数（`#5940`）**：
   - 将遍历 `self.storage_backends` 的 `backend.batched_put(...)` 包裹在 `try...finally` 中，无论任意后端同步抛错，均在 `finally` 中对整批 `memory_objs` 执行 `ref_count_down()`，实现 **`0` 内存块泄漏**；同时在 `#5696` 中令 `is_scratch_spec` 兼容旧版 vLLM 的 `participates_in_prefix_caching=False`。
2. **Mooncake `CursorSlice{index, count}` 与 `ImmutableBucketAllocator::DeferDeletion`（`#4854, #4858`）**：
   - `ReadCursor(tenant_id, CursorSlice{index, count})` 将单个租户的对象集合划分为 `count` 个互不相交的切片（对任意 `count in {1, 3, 16}` 形成严格双射覆盖），使多线程可并行扫描租户的 in-flight 任务与过期副本。
   - `DeferDeletion(std::move(*logical), not_before)` 将已被逻辑淘汰的 DFS 桶放入 `waiting_deletions_` 映射，直到系统时钟越过活跃读租约截止时间 `not_before` 后才由 `RetryFailedEvictions()` 物理删除底层文件。
   - `#4858` 新增 `POST /api/v1/soft_pin` 与 `/api/v1/soft_unpin`（单批最多 `4,096` 个 key），并将 `max_kv_soft_pin_ttl` 从 `24h`（`86,400,000 ms`）提升至 **`7d`（`604,800,000 ms`）**。

---

## 3. 架构师设计练习与可证伪问题

1. **定量练习**：对于 `S*B = 8192, V_local = 131072, H = 4096, K = 3` 的 MTP 训练配置，验证延迟反向传播保留 4 个 `[S, B, V]` FP32 张量共 `16.0 GiB`，而 `mtp_loss_early_backward=True` 峰值仅需 `1 * 4.0 GiB + 3 * 64 MiB = 4.1875 GiB`（节省 `73.83%`）。
2. **可证伪假设**：在 Mooncake `ImmutableBucketAllocator` 中，调用 `DeferDeletion(bucket, now + 200ms)` 后立即调用 `RetryFailedEvictions()` 返回 `0` 且文件仍在磁盘上；等待 `200ms` 后再次调用则返回 `1` 并完成物理删除。

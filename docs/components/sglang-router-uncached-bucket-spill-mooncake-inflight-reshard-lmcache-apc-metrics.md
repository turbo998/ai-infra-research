# SGLang 未缓存词元 Prefill 分桶路由与空闲 DP 溢出、Mooncake 有界在途权重重分片与 LMCache APC 前缀剔除命中率指标

> **English Version**: [sglang-router-uncached-bucket-spill-mooncake-inflight-reshard-lmcache-apc-metrics.en.md](sglang-router-uncached-bucket-spill-mooncake-inflight-reshard-lmcache-apc-metrics.en.md)  
> **证据等级**: `implementation`（源码审计）+ `own-offline`（确定性离线建模）  
> **对应实验**: [`experiments/sglang_uncached_bucket_mooncake_inflight_lmcache_apc/run.py`](../../experiments/sglang_uncached_bucket_mooncake_inflight_lmcache_apc/run.py)

---

## 1. 工程背景与核心问题

在 PD 分离（Prefill-Decode Disaggregation）、在线权重重分片（Online Weight Resharding）与多级 KV 缓存（Engine APC + LMCache L1/L2）协同架构中，以下三个机制决定了集群尾延迟与可观测性精度：
1. **SGLang Router 按总输入长度分桶导致的 Warm 长上下文误路由与桶拥塞（`sgl-project/sglang@df13d5dc`，`#45103`）**：当长上下文请求（如 `L=20,000`）在 KV Indexer 中命中 `15/20` 个前缀块（可复用 `15,000` 词元）时，实际需要 Prefill 扩展计算的未缓存词元仅 `E = 5,000`；若仍按 `L=20,000` 路由到 `long` 桶（`min_extend_tokens: 8193`），不仅浪费长桶算力，且当目标桶所有 DP rank 均在排队或执行分块 Prefill 时无法利用其他桶的空闲 DP rank。
2. **Mooncake 权重重分片同步逐批排空（Window=1）的吞吐受限与异常生命周期风险（`kvcache-ai/Mooncake@0977fc70`，`#4628`）**：此前 `mooncake-reshard` 每提交一个批次必须同步 `drain` 至终态才能发起下一批；若引入多批次重叠（`max_inflight_batches=N`），必须保证多个在途 `ScatterTransferTicket` 共享统一超时预算，且不能因异常丢弃未完成 ticket 引用而触发持有 Python GIL 的 C++ 析构阻塞。
3. **LMCache 将引擎 APC 已覆盖前缀计入 L1 命中率导致指标虚高（`LMCache/LMCache@168130df`，`#5836`）**：当引擎自身的自动前缀缓存（APC）已命中前 `start_chunk` 个块并通过 `key.start` 跳过查询时，旧版 `_consume_prefetch_result` 执行了 `l1_found_count += job.start_chunk` 且 `requested_tokens = len(chunk_hashes) * chunk_size`，导致 APC 已持有的前缀被错误归因为 LMCache L1 命中。

---

## 2. 一手源码机制剖析与公式推导

### 2.1 SGLang `--route-by-uncached-tokens` 与 `IdleSpill` 决策状态机
在 `experimental/sgl-router/src/policies/selection.rs` 与 `buckets.rs` 中：
- **未缓存词元 $E$ 计算与桶匹配**：
  $$E = \max\bigl(0,\; L_{\text{seq}} - \max_{w \in \text{workers}} \text{estimate\_matched\_prefix\_tokens}(L_{\text{seq}}, B_{\text{query}}, B_{\text{matched}}(w))\bigr)$$
  分桶兼容性 `prefill_compatible(spec, extend_tokens, input_tokens)` 同时校验 `extend_tokens` 落在 `[min_extend_tokens, max_extend_tokens]` 内，且完整上下文 `input_tokens <= max_context_tokens`（若无桶容纳 `(E, L)`，则自动回退至按 `input_tokens` 匹配）。
- **冷请求（`reusable == 0`）跨域汇聚**：当 `route_by_uncached_tokens=True` 且无前缀命中（`uncached_tokens = None`）时，调用 `prefill_domains_for_any_extend` 汇聚所有 `max_context_tokens` 兼容的 Prefill worker（去重后组成全局候选集），直接按最小负载路由。
- **跨桶空闲 DP Rank 溢出（`CacheAwareDecision::IdleSpill`）**：在 `engine_reported_load.rs` 中，每个 DP rank 的待处理 Prefill 负载定义为 `pending(r) = num_waiting_reqs(r) + num_waiting_uncached_tokens(r)`（包含分块 Prefill 尚未算完的剩余词元），worker 的最小 rank 待处理工作量为 `min_rank_pending(w) = min_r pending(r)`。当 `E` 所属桶内**所有**候选 worker 的 `min_rank_pending(w) > 0` 时，`idle_spill_domains()` 筛选其他上下文兼容桶中满足 `min_rank_pending(w) == 0`（至少有一个完全空闲 DP rank）的 worker 并触发 `idle_spill` 路由。

### 2.2 Mooncake `max_inflight_batches` 滑动窗口与 `_CompositeCompletionTicket`
在 `mooncake-reshard/python/mooncake/reshard/transfer_engine/{executor.py,completion.py}` 中：
- **FIFO 滑动窗口提交**：通过原生非阻塞 `submit_scatter_read` / `submit_scatter_write` 提交批次并将 `_PublishedBatch` 压入 `deque`；当 `len(published) >= max_inflight_batches` 时弹出队首 `popleft()` 并排空。在 8 个批次（每次提交开销 `2 ms`、RDMA 传输 `12 ms`）基准下，`max_inflight_batches=1` 总耗时 `8 * 14 = 112.0 ms`，而 `max_inflight_batches=4` 缩短至 `34.0 ms`（节省 `69.64%`）。
- **单调时钟共享截止期（Composite Deadline）与 GIL 隔离**：当多个在途 ticket 处于 `COMPLETION_UNKNOWN` 时，封装为 `_CompositeCompletionTicket`，在 `drain(timeout_ms)` 中按 `deadline_ns = monotonic_ns() + timeout_ms * 10^6` 计算每个子 ticket 的剩余预算 `remaining_ms = max(0, (deadline_ns - now_ns) // 10^6)`（如总预算 `1000 ms`、子耗时 `[400, 350, 500] ms` 时依次传入 `[1000, 600, 250] ms`，总等待严格受限于 `1000 ms`）。若 ticket 属性损坏无法排空，`_UndrainableCompletionUnknownTicket(owner=ticket)` 强引用保留底层 C++ 对象，防止 Python GC 触发持有 GIL 的原生析构死锁。

### 2.3 LMCache APC 覆盖前缀剔除与双视角命中率公式
在 `lmcache/v1/multiprocess/modules/lookup.py` 中，令 `start_chunk = min(max(0, key.start // chunk_size), len(chunk_hashes))`：
- **LMCache 自身查找区间指标（剔除 APC 前缀）**：
  $$\text{apc\_covered\_tokens} = \text{start\_chunk} \times C_{\text{chunk}}, \quad \text{requested\_tokens} = N_{\text{chunks}} C_{\text{chunk}} - \text{apc\_covered\_tokens}$$
  $$\text{l1\_hit\_tokens} = \min(c_{\text{L1}}, c_{\text{found}}) \times C_{\text{chunk}}, \quad \text{hit\_tokens} = (\text{l1\_chunks} + \text{l2\_chunks}) \times C_{\text{chunk}}$$
- **请求级全 Prompt 命中率指标（Engine APC + LMCache 联合覆盖）**：
  $$\text{request\_hit\_tokens} = \max\bigl(\text{engine\_hit\_tokens},\; \text{apc\_covered\_tokens} + \text{hit\_tokens}\bigr), \quad \text{request\_hit\_rate} = \frac{\text{request\_hit\_tokens}}{\text{prompt\_tokens}}$$
  取 `max` 而非相加是因为 `engine_hit_tokens`（未对齐块边界的引擎提示值）与 `apc_covered_tokens` 均从位置 0 开始，在 `key.start` 以下的子块尾部区间存在重叠。

---

## 3. 架构师设计练习与可证伪问题

1. **计算题**：请求总长 `prompt_tokens = 2100`，`chunk_size = 256`（共 8 个完整块 = `2048` 词元），引擎报告 `engine_computed_tokens = 600`，`key.start = 512`（`start_chunk = 2`），LMCache 在 `[2, 8)` 区间命中 `2` 个 L1 块与 `3` 个 L2 块。求 `requested_tokens`、`l1_hit_tokens`、`hit_tokens` 与 `request_hit_tokens`。
   - **解答**：`apc_covered_tokens = 2 * 256 = 512`；`requested_tokens = 2048 - 512 = 1536`；`l1_hit_tokens = 2 * 256 = 512`（旧版误算为 `1024`）；`hit_tokens = (2 + 3) * 256 = 1280`；`request_hit_tokens = max(600, 512 + 1280) = 1792`（请求级命中率 `1792 / 2100 = 85.33%`）。
2. **可证伪假设**：在 `sgl-router` 中，若 `short` 桶某个 DP rank 的排队请求数 `num_waiting_reqs == 0`，但正在执行分块 Prefill（`num_waiting_uncached_tokens = 40,000`），繁忙的 `long` 桶 Warm 请求是否会向该 rank 触发 `idle_spill`？（不会，因为 `min_rank_pending = 0 + 40,000 > 0`，该 rank 并非空闲）。

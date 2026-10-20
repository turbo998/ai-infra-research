# SGLang `KVShard` 页交错 PD 分离传输、`sgl-router` 未缓存 Token 分桶路由与 Qwen3 MoE EPLB 重平衡保护

> **English Version**: [sglang-kvshard-pd-transfer-uncached-bucket-routing.en.md](sglang-kvshard-pd-transfer-uncached-bucket-routing.en.md)  
> **证据等级 (Evidence Class)**: `implementation`（一手源码核验） + `own-offline`（确定性标准库仿真）  
> **上游仓库与提交**: `sgl-project/sglang` commits [`eb4b167d2749840eae18167fe9e0a608308e45aa`](https://github.com/sgl-project/sglang/commit/eb4b167d2749840eae18167fe9e0a608308e45aa), [`df13d5dce31abfeff0e701a62e8c39a136758e6c`](https://github.com/sgl-project/sglang/commit/df13d5dce31abfeff0e701a62e8c39a136758e6c) & [`9db58ab547516435b544ae050dd6c2f91b52bcf3`](https://github.com/sgl-project/sglang/commit/9db58ab547516435b544ae050dd6c2f91b52bcf3) (2026-10-20)  
> **关联实验**: [`experiments/sglang_kvshard_uncached_bucket_sim/run.py`](../../experiments/sglang_kvshard_uncached_bucket_sim/run.py) (`--check` 通过)

---

## 1. 架构背景与核心工程矛盾

1. **MLA 预填充逻辑页分片 (`KVShard`) 下的 PD 传输难题**：在 MLA 架构中，为了把 Prefill 阶段的 KV 缓存容量扩展 $N$ 倍（`kv_shard_size = N`），`enable_kv_cache_sharding` 将逻辑 KV 页按模 $N$ 交错存储在不同的 Prefill Attention-TP Rank 上（没有任何单个 Rank 持有完整页序列）。传统 PD 传输假设单个 Prefill Rank 拥有连续的 `index_slice`，直接传输会导致页缺失或错位；同时，最坏分片下的页对齐余量若未在 Bootstrap 准入时扣除，会在 `add_one_req_ignore_eos` 阶段触发单分片物理池溢出。
2. **长 Prompt 高前缀命中率下的分桶错配**：当 `sgl-router` 将 Prefill 实例划分为 `short`（`max_extend_tokens = 8,192`）与 `long`（`min_extend_tokens = 8,193`）分桶时，一个 `20,000` Token 的请求若在缓存中命中了 `15,000` Token，其实际需要 Prefill 计算的未缓存工作量仅为 `E = 5,000` Token。若按原始输入长度路由，该请求仍被发往高负载的 `long` 分桶，浪费 `short` 分桶算力。
3. **Qwen3 MoE 在 EPLB 动态重平衡后的静默输出损坏**：开启 `enable_eplb` 后，专家权重在物理槽位间发生重排，但 `Qwen3MoeSparseMoeBlock.forward_normal` 调用 `self.topk(...)` 时漏传 `expert_location_dispatch_info`，导致逻辑专家 ID 未转换为重排后的物理专家槽位。

---

## 2. 一手源码机制拆解 (`sgl-project/sglang`)

### 2.1 `KVShard` 页交错过滤、全 TP 拉取与最严 Bootstrap 准入上界 (`conn.py`, `prefill.py`, `utils.py` @ `eb4b167d27`)

- **模 $N$ 本地页映射 (`filter_kv_indices_for_shard_rank`)**：
  ```python
  owned = kv_indices % shard_size == shard_rank
  positions = np.arange(index_slice.start, index_slice.stop, dtype=np.int64)
  return kv_indices[owned] // shard_size, positions[owned]
  ```
  逻辑页 $l$ 存储在分片 Rank $r = l \bmod N$ 的本地物理页 $\lfloor l / N \rfloor$ 上，并用稀疏的 `positions[owned]` 替代原先连续的 `index_slice`。当某个非末尾分块（`not is_last_chunk`）在当前 Rank 上过滤后 `len(kv_indices) == 0` 时，直接跳过 RDMA 发送。
- **全 Attention-TP 扇入 (`_resolve_rank_mapping`)**：当 `is_mla_backend and info.enable_kv_cache_sharding and info.attn_cp_size == 1` 时，强制要求 Prefill 与 Decode 的 `attn_tp_size` 相等，并将 `target_tp_ranks = list(range(info.attn_tp_size))`、`required_dst_info_num = self.attn_tp_size`，使每个 Decode Rank 从全部 $N$ 个 Prefill TP Rank 收集各自持有的 `1/N` 页。
- **最严分片 Bootstrap 准入公式 (`max_bootstrap_input_len`)**：
  $$L_{\max} = \left\lfloor \frac{\text{capacity} - N \cdot P - 3}{P} \right\rfloor \cdot P$$
  其中 $\text{capacity} = \text{per\_rank\_tokens} \times N$，扣除每分片最多 1 页的向上取整损耗 $N \cdot P$ 以及 `add_one_req_ignore_eos` 所需的 1 个新生成 Token、1 个备用 Token 与严格小于号边界（共 `- 3`）。在 `per_rank_tokens = 16,384, P = 64, N = 4` 下，$\text{capacity} = 65,536$，而 $L_{\max} = 65,216$（提前拦截 `[65,217, 65,536]` 区间内会导致最坏分片溢出的请求）。

### 2.2 `sgl-router` 未缓存 Token (`E`) 分桶路由与空闲 DP Rank 溢出 (`buckets.rs`, `selection.rs` @ `df13d5dce3`)

1. **最长可复用前缀估算 (`best_reusable_tokens`)**：遍历候选 Worker 在 `sgl_kv_indexer` 中的前缀命中块数，计算最大可复用 Token 数 `reusable`，并定义未缓存扩展长度：
   $$E = \text{request\_sequence\_tokens} - \text{reusable}$$
2. **双重兼容性谓词 (`prefill_compatible`)**：
   $$\text{compatible}(\text{spec}, E, L) \iff E \in [\text{min\_extend}, \text{max\_extend}] \;\land\; L \le \text{max\_context\_tokens}$$
   即使 $E = 5,000 \le 8,192$ 落入 `short` 桶的扩展区间，若 `short` 桶配置了 `max_context_tokens = 10,000 < L = 20,000`，`prefill_compatible` 也会拒绝 `short` 并安全回退至按输入长度匹配的 `long` 桶。
3. **空闲 DP Rank 溢出 (`idle_spill_domains` -> `CacheAwareDecision::IdleSpill`)**：
   - 当承载 $E$ 的目标分桶内所有 Worker 的每个 DP Rank 均有未完成工作（`fresh_min_rank_pending_for_url > 0`，其中任意排队的请求 `waiting > 0` **或**正在分块预填充的未缓存 Token `num_waiting_uncached_tokens > 0` 均视为非空闲）时，路由器从满足上下文上限（`prefill_compatible(spec, None, L)`）的其他分桶中筛选出拥有完全空闲 DP Rank（`waiting == 0 && uncached == 0`）的 Worker 进行溢出调度。

### 2.3 Qwen3 MoE EPLB 专家位置分发信息透传 (`qwen3_moe.py` @ `9db58ab547`)

在 `Qwen3MoeSparseMoeBlock.forward_normal` 中构造 `dispatch_info = ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id) if get_exec().moe.enable_eplb else None` 并传入 `self.topk(hidden_states, router_logits, expert_location_dispatch_info=dispatch_info)`，确保 Top-K 选出的逻辑专家编号在送入 `self.experts(...)` 前完成 `logical -> physical` 重映射。

---

## 3. 确定性离线实验验证 (`sglang_kvshard_uncached_bucket_sim`)

运行 `python experiments/sglang_kvshard_uncached_bucket_sim/run.py --check`，核心验证结果如下：

| 验证场景 | 关键参数 | 传统策略 / 修复前行为 | SGLang 新机制输出 |
|---|---|---|---|
| `KVShard` 4 分片 PD 传输 | `N=4, P=64`, 单卡 `16,384` Token (`capacity=65,536`) | 未扣除分片取整余量（`65,536` 误准入溢出）；空分块仍发信令 | `max_bootstrap_input_len = 65,216`（预留 `320` Token）；跳过 **`3` 个**非末尾空分块；12 页 `100%` 无损重建 |
| 热请求未缓存路由 (`20K` Prompt) | 命中 `15K` Token (`E = 5,000`)，`short_max_ctx = None` | 按总长 `20,000` 路由至 `long` | 按 `E = 5,000 <= 8,192` 路由至 **`short`** (`mode = warm`) |
| 上下文上限保护 (`20K` Prompt) | `E = 5,000`，但 `short_max_ctx = 10,000` | 若仅按 `E` 路由会违反 `short` 显存上限 | `prefill_compatible` 拦截并回退至 **`long`** |
| 空闲 DP Rank 溢出 (`IdleSpill`) | `E = 15,000` (`long` 忙 `[(2,9)]`)，`short` 含空闲 Rank `[(3,9), (0,0)]` | 死等 `long` 队列 | 触发 `idle_spill` 溢出至 **`short`**；当 `short` Rank 处于分块预填充 `(0, 40000)` 时拒绝误溢出 |
| Qwen3 MoE EPLB 重平衡 | `topk_logical = [0, 3, 7]`, 物理槽位置换 `[5,2,7,0,6,1,3,4]` | 漏传 `dispatch_info` 输出 `190.0`（权重错位损坏） | 传入 `ExpertLocationDispatchInfo` 输出 **`120.0`**（与理论值 `120.0` 完全一致） |

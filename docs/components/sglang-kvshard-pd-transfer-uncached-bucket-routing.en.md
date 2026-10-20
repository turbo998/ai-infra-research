# SGLang `KVShard` Page-Interleaved PD Transfer, `sgl-router` Uncached-Token Bucket Routing & Qwen3 MoE EPLB Integrity

> **中文版**: [sglang-kvshard-pd-transfer-uncached-bucket-routing.md](sglang-kvshard-pd-transfer-uncached-bucket-routing.md)  
> **Evidence Class**: `implementation` (First-hand source verification) + `own-offline` (Deterministic stdlib simulation)  
> **Upstream Repository & Commits**: `sgl-project/sglang` commits [`eb4b167d2749840eae18167fe9e0a608308e45aa`](https://github.com/sgl-project/sglang/commit/eb4b167d2749840eae18167fe9e0a608308e45aa), [`df13d5dce31abfeff0e701a62e8c39a136758e6c`](https://github.com/sgl-project/sglang/commit/df13d5dce31abfeff0e701a62e8c39a136758e6c) & [`9db58ab547516435b544ae050dd6c2f91b52bcf3`](https://github.com/sgl-project/sglang/commit/9db58ab547516435b544ae050dd6c2f91b52bcf3) (2026-10-20)  
> **Experiment**: [`experiments/sglang_kvshard_uncached_bucket_sim/run.py`](../../experiments/sglang_kvshard_uncached_bucket_sim/run.py) (`--check` passed)

---

## 1. Architectural Motivation & Engineering Bottlenecks

1. **Page-Interleaved MLA KV Sharding (`KVShard`) in PD Disaggregation**: To widen prefill KV capacity by $N\times$ (`kv_shard_size = N`), `enable_kv_cache_sharding` distributes logical KV pages modulo $N$ across prefill attention-TP ranks. Because no single rank holds a full contiguous slice of pages, legacy 1-to-1 TP rank transfers drop `(N-1)/N` of the pages, and unguarded bootstrap admission can overflow individual shards due to per-shard page rounding.
2. **Bucket Misrouting on Cache-Warm Long Prompts**: When `sgl-router` splits prefill workers into `short` (`max_extend_tokens = 8,192`) and `long` (`min_extend_tokens = 8,193`) buckets, a `20,000`-token prompt that hits `15,000` cached prefix tokens only needs `E = 5,000` uncached extend tokens. Routing by raw input length forces it onto busy `long` workers while `short` workers sit idle.
3. **Post-EPLB MoE Output Corruption in Qwen3 MoE**: When `enable_eplb` relocates hot experts across physical slots, `Qwen3MoeSparseMoeBlock.forward_normal` omitted `expert_location_dispatch_info` in `self.topk(...)`, leaving logical expert IDs unmapped to physical slots.

---

## 2. First-Hand Source Analysis (`sgl-project/sglang`)

### 2.1 `KVShard` Modulo-$N$ Page Filtering, All-TP Fan-In & Strict Bootstrap Admission (`conn.py`, `prefill.py`, `utils.py` @ `eb4b167d27`)

- **Modulo-$N$ Local Page Mapping (`filter_kv_indices_for_shard_rank`)**:
  Logical page $l$ resides on shard rank $r = l \bmod N$ as local page $\lfloor l / N \rfloor$. `filter_kv_indices_for_shard_rank` returns `(kv_indices[owned] // shard_size, positions[owned])`, replacing the contiguous `index_slice` with exact sparse send positions, and skips non-final chunks when `len(kv_indices) == 0`.
- **All-TP Fan-In (`_resolve_rank_mapping`)**: Under `is_mla_backend and info.enable_kv_cache_sharding and info.attn_cp_size == 1`, every decode rank pulls from all `info.attn_tp_size` prefill ranks (`required_dst_info_num = self.attn_tp_size`).
- **Worst-Case Sharded Admission Bound (`max_bootstrap_input_len`)**:
  $$L_{\max} = \left\lfloor \frac{\text{capacity} - N \cdot P - 3}{P} \right\rfloor \cdot P$$
  Subtracting $N \cdot P$ (one page-rounding slack per shard) plus `3` tokens (`1` new token + `1` spare token + strict inequality in `add_one_req_ignore_eos`) guarantees zero per-shard overflow. At `16,384` physical tokens/rank (`P = 64, N = 4`), `capacity = 65,536` and `max_input_len = 65,216` tokens.

### 2.2 `sgl-router` Uncached-Token (`E`) Bucket Routing & Idle DP-Rank Spill (`buckets.rs`, `selection.rs` @ `df13d5dce3`)

1. **Longest Reusable Prefix (`best_reusable_tokens`)**: Computes `uncached_tokens = E = request_sequence_tokens.saturating_sub(reusable)` across candidate workers in `sgl_kv_indexer`.
2. **Dual Compatibility Predicate (`prefill_compatible`)**:
   $$\text{compatible}(\text{spec}, E, L) \iff E \in [\text{min\_extend}, \text{max\_extend}] \;\land\; L \le \text{max\_context\_tokens}$$
   If `E = 5,000` fits `short` (`<= 8,192`) but `short.max_context_tokens = 10,000 < L = 20,000`, `prefill_compatible` rejects `short` and falls back to `long`.
3. **Idle DP-Rank Spill (`idle_spill_domains` -> `CacheAwareDecision::IdleSpill`)**: When every DP rank of every worker in $E$'s bucket has pending work (`fresh_min_rank_pending_for_url > 0`, counting both queued requests `num_waiting_reqs > 0` and in-flight chunked prefill `num_waiting_uncached_tokens > 0`), the selector spills to any context-compatible bucket worker possessing an idle DP rank (`waiting == 0 && uncached == 0`).

### 2.3 Qwen3 MoE EPLB Dispatch Info Propagation (`qwen3_moe.py` @ `9db58ab547`)

`Qwen3MoeSparseMoeBlock.forward_normal` constructs `dispatch_info = ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id) if get_exec().moe.enable_eplb else None` and passes it to `self.topk(...)` so logical expert indices are translated to rebalanced physical expert slots before `self.experts(...)`.

---

## 3. Deterministic Offline Verification (`sglang_kvshard_uncached_bucket_sim`)

Running `python experiments/sglang_kvshard_uncached_bucket_sim/run.py --check` verifies:

| Scenario | Parameters | Legacy Behavior | SGLang Verified Outcome |
|---|---|---|---|
| `KVShard` 4-Way PD Transfer | `N=4, P=64`, `16,384` tokens/rank (`capacity=65,536`) | Admits up to `65,536` (shard overflow); sends empty chunks | `max_bootstrap_input_len = 65,216` (`320`-token guard); skips **`3`** empty non-final chunks; 12 pages reconstructed losslessly |
| Warm Uncached Routing (`20K` prompt) | `reusable = 15,000` (`E = 5,000`), `short_max_ctx = None` | Routed to `long` by raw `20,000` length | Routed to **`short`** (`E = 5,000 <= 8,192`, `mode = warm`) |
| Context-Cap Guard (`20K` prompt) | `E = 5,000`, `short_max_ctx = 10,000` | Context overflow if routed by `E` alone | Rejected by `short_max_ctx`; safely falls back to **`long`** |
| Idle DP-Rank Spill (`IdleSpill`) | `E = 15,000` (`long` busy `[(2,9)]`), `short` has `[(3,9), (0,0)]` | Queues behind busy `long` worker | Spills to **`short`** (`idle_spill`); blocks spill when `short` rank is chunk-prefilling `(0, 40000)` |
| Qwen3 MoE EPLB Rebalance | `topk_logical = [0, 3, 7]`, perm `[5,2,7,0,6,1,3,4]` | Output corrupted (`190.0` vs `120.0`) | `ExpertLocationDispatchInfo` restores exact **`120.0`** output |

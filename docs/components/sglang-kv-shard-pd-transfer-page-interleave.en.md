# SGLang Logical-Page KV Cache Sharding (`KVShard`) & PD Disaggregated Transfer Contract: `filter_kv_indices_for_shard_rank` Modulo-Quotient Remapping, `max_bootstrap_input_len` Admission Boundary & All-Shard Pull Topology

> **中文版**: [sglang-kv-shard-pd-transfer-page-interleave.md](sglang-kv-shard-pd-transfer-page-interleave.md)  
> **Evidence Class**: `implementation` (verified against upstream commits) + `own-offline` (`experiments/sglang_kv_shard_pd_transfer/run.py`)

---

## 1. Architectural Problem Statement

In Multi-head Latent Attention (**MLA**) models such as DeepSeek-V3/R1, standard attention tensor parallelism replicates the compressed latent KV cache across every TP rank. `sgl-project/sglang` introduces logical-page interleaved KV sharding (`--enable-kv-cache-sharding`, `KVShard`), where logical page $l$ resides exclusively on prefill shard rank $r = l \bmod N$ as local physical page $l_{\text{local}} = \lfloor l / N \rfloor$, widening aggregate prefill KV capacity by $N\times$. Integrating `KVShard` with **Prefill/Decode (P/D) disaggregation** requires solving three core protocol hazards:

1. **Non-Congruent Page Chains & Empty Chunks**: Under chunked prefill and `UnifiedRadixCache` rotated page chains, owned pages are non-contiguous. Using a contiguous `index_slice` or sending empty non-final chunks corrupts Decode KV assembly.
2. **All-Shard MLA Pull & DCP Mutual Exclusion**: Because no single prefill TP rank holds a full KV copy, every decode rank must pull from all $N$ prefill TP ranks (`target_tp_ranks = [0..N-1]`), and Decode Context Parallelism (`DCP`) must be explicitly rejected when `--enable-kv-cache-sharding` is active.
3. **Bootstrap Queue Tail Deadlock (`300 s` Decode Timeout)**: The strictest sharded admission path (`add_one_req_ignore_eos`) requires page-rounded prompt tokens plus one page per shard plus 2 tokens to stay below `capacity`. Admitting prompts up to raw `capacity` traps tail prompts (`[limit+1 .. capacity]`) forever in the prefill bootstrap queue until Decode hits its `300 s` KV-wait timeout.

---

## 2. First-Hand Upstream Implementation Verification (`implementation`)

### 2.1 `filter_kv_indices_for_shard_rank` & Non-Final Empty Chunk Skip (`eb4b167d27`, `#38473`)

In `python/sglang/srt/disaggregation/utils.py` (blob `44a4f111c24b226349fc6b6510a8f919dcf8e792`) and `python/sglang/srt/disaggregation/common/conn.py` (blob `7b6012eebe29f3f7707c32361ecc5e310a5f6463`):
- `filter_kv_indices_for_shard_rank` filters `owned = (kv_indices % shard_size == shard_rank)` and returns `(kv_indices[owned] // shard_size, positions[owned])`, replacing the contiguous `index_slice` with exact request-global send positions.
- In `CommonKVSender._prepare_send_indices`, if `len(kv_indices) == 0 and not is_last_chunk`, the sender skips the RDMA transfer for that chunk while preserving final-chunk completion signaling (`is_last_chunk == True`).

### 2.2 MLA All-Shard Rank Mapping & DCP Guard (`conn.py` blob `7b6012eebe`)

- `CommonKVManager.try_ensure_parallel_info` raises `RuntimeError("PD decode DCP does not support a prefill with --enable-kv-cache-sharding.")` when `self.dcp_size > 1` and `info.enable_kv_cache_sharding` is true.
- `CommonKVManager._resolve_rank_mapping` checks `self.is_mla_backend and info.enable_kv_cache_sharding and info.attn_cp_size == 1`, asserts `self.attn_tp_size == info.attn_tp_size`, and configures `target_tp_rank = None`, `target_tp_ranks = list(range(info.attn_tp_size))`, `required_dst_info_num = self.attn_tp_size`, and `required_prefill_response_num = info.attn_tp_size`.

### 2.3 `max_bootstrap_input_len` Admission Formula & Dual-Needle E2E Verification (`prefill.py` blob `8c5fd2c374` & `test_disaggregation_kv_cache_sharding.py` blob `11fc1cf565`)

In `python/sglang/srt/disaggregation/prefill.py`:

$$\text{max\_bootstrap\_input\_len}(C, P, N) = \begin{cases} C, & N = 1 \\ \left\lfloor \dfrac{C - N \cdot P - 3}{P} \right\rfloor \cdot P, & N > 1 \end{cases}$$

`PrefillBootstrapQueue._check_if_req_exceed_kv_capacity` enforces `len(req.origin_input_ids) <= self.max_input_len`, immediately returning `HTTP 400 BAD_REQUEST` instead of stalling Decode for `300 s`.

---

## 3. Deterministic Standard-Library Verification (`own-offline`)

Run `python experiments\sglang_kv_shard_pd_transfer\run.py --check`:

| Shard Size $N$ | Physical Tokens / Rank ($C_{\text{rank}}$) | Widened Logical Capacity ($C$) | `max_bootstrap_input_len` | Rejected Unschedulable Tail ($C - \text{limit}$) | Prevented Decode Timeout |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `1` | `16,384` | `16,384` | **`16,384`** | `0` tokens | `0 s` |
| `2` | `16,384` | `32,768` | **`32,576`** | **`192` tokens** | **`300 s`** |
| `4` | `16,384` | `65,536` | **`65,216`** | **`320` tokens** | **`300 s`** |
| `8` | `16,384` | `131,072` | **`130,496`** | **`576` tokens** | **`300 s`** |

- **Rotated Chain Reconstruction**: Across a 9-page rotated chain `[2, 7, 8, 13, 18, 23, 24, 29, 34]` at `slice(20, 29)` with $N=4$, `filter_kv_indices_for_shard_rank` produces disjoint local pages and send positions that reconstruct **100% bit-identically** on Decode.
- **Dual-Needle 1.50x Pool Verification**: A `24,576`-token prompt (`1.50x` a single rank's `16,384`-token pool) places the `red` needle at page `20` (rank `0`, local page `10`) and the `blue` needle at page `341` (rank `1`, local page `170`), recovering both pass keys on Decode.

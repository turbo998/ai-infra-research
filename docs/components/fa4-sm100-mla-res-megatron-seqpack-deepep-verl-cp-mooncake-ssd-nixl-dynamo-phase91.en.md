# Phase 91 Track C: FlashAttention-4 SM100 Sparse MLA O-Residual Guard & ROCm/CK Split-KV Heuristic Fix, Megatron-LM Sequence-Packing Middle-PP Guard, DeepEP Internode Combine Proxy Fence, verl Teacher CP Alignment, Mooncake SSD Prefetch & Weight Lifecycle, NIXL Bidirectional NVTX Correlation & Dynamo KV-Router Scalar Walk

> **中文版本**: [fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91.md](fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91.md)  
> **Phase Status**: Phase 91 (`priority: 100 — 100-Priority Milestone!`) · Question `q-fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91` · Claim `claim-fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91`  
> **Reproducible Experiment**: [`experiments/fa4_sm100_mla_res_megatron_seqpack_deepep_verl_cp_mooncake_ssd_nixl_dynamo_phase91/`](../../experiments/fa4_sm100_mla_res_megatron_seqpack_deepep_verl_cp_mooncake_ssd_nixl_dynamo_phase91/)

---

## 1. First-Hand Source Evidence Matrix (10 Upstream Commits)

| # | Repository & Commit | Date | Evidence Class | Core Architectural Mechanism |
|---|---|---|---|---|
| 1 | `Dao-AILab/flash-attention` [`2cb0a8e471`](https://github.com/Dao-AILab/flash-attention/commit/2cb0a8e471b837338275b56fc8dc792cc0bf6070) | 2026-09-23 | `implementation` | SM100 CuTe sparse-MLA forward predicates the register-to-global `o_lo` residual store with `is_valid_qhead_row` when `real_q_heads < 128`, preventing overwrite of the next token's `o_lo` or out-of-bounds memory access |
| 2 | `Dao-AILab/flash-attention` [`0b4cb4e64b`](https://github.com/Dao-AILab/flash-attention/commit/0b4cb4e64b0dd53a6042a596844f692aa5b473ab) (`#2901`) | 2026-09-24 | `implementation` + `author-benchmark` | Fixes ROCm/CK `override_num_splits_if_necessary` (which was pinned at `num_splits = 1` due to `num_n_blocks = 1` and `nhead` vs `nhead_k`) and caps `num_splits <= 8` (`3.21x` speedup at `batch*nhead_k = 64`) |
| 3 | `NVIDIA/Megatron-LM` [`ede2c586cf`](https://github.com/NVIDIA/Megatron-LM/commit/ede2c586cf0f8e1a26c8ca7fc9fa7d7dfea83c7a) (`#8378`) | 2026-10-21 | `implementation` | `is_dataset_built_on_rank` returns `False` on middle pipeline stages when `sequence_packing_scheduler` is enabled |
| 4 | `deepseek-ai/DeepEP` [`d4f41e4e93`](https://github.com/deepseek-ai/DeepEP/commit/d4f41e4e937d99fd262ca53f5e4221dd17fd5086) (`#642`) | 2026-05-26 | `implementation` | Inserts `fence_view_async_shared()` (`fence.proxy.async.shared::cta`) before `mbarrier_arrive(empty_barriers[stage_idx])` in `internode_ll.cu` low-latency combine receive |
| 5 | `volcengine/verl` [`67bf1df3f2`](https://github.com/volcengine/verl/commit/67bf1df3f205ab06fd73915049f180b183dc8771) (`#8332`) | 2026-10-22 | `implementation` | `compute_forward_kl_topk` packs teacher top-k tensors with the student's `cp_layout` (`dsv4_hybrid` contiguous vs `zigzag`) and `local_cp_size`, eliminating up to 100% row misalignment on CP=2 THD |
| 6 | `kvcache-ai/Mooncake` [`f751cff2b6`](https://github.com/kvcache-ai/Mooncake/commit/f751cff2b698a9df2c9b3efbfdc36ca765656df9) (`#4272`) | 2026-10-26 | `implementation` | Mooncake Store adds lease-free `QueryReadOnly` and background SSD-to-DRAM promotion on `is_exist` (`ExistOptions.prefetch_to_memory = true`) |
| 7 | `kvcache-ai/Mooncake` [`2e533c97fe`](https://github.com/kvcache-ai/Mooncake/commit/2e533c97fe93ee916967a7e29a9f5590c82e7e7a) (`#4313`) | 2026-10-26 | `implementation` | Mooncake Store adds HA-replicated managed weight group revision lifecycle RPCs and protects managed weight groups from generic cache eviction |
| 8 | `ai-dynamo/nixl` [`a95227d202`](https://github.com/ai-dynamo/nixl/commit/a95227d20229df99b0363042d37c3de75ffdf378) (`#2495`) | 2026-10-26 | `implementation` | NIXL sets `nixl.xfer_request_id` as the `uint64` NVTX payload on both sender (`postXferReq`, `nixl::xfer.complete`) and receiver (`nixl::notif.received`) |
| 9 | `ai-dynamo/dynamo` [`75d0686fb7`](https://github.com/ai-dynamo/dynamo/commit/75d0686fb7f07a5fb9665e33d8723cb0cb77a78e) (`#16765`) | 2026-10-23 | `implementation` | Dynamo SGLang `native_http.rs` moves `response` directly into `Map::new()` without cloning large `routed_experts` payloads |
| 10 | `ai-dynamo/dynamo` [`30e972003a`](https://github.com/ai-dynamo/dynamo/commit/30e972003a0e4fa3577cf899848b715bd4c97797) (`#17149`) | 2026-10-23 | `implementation` | Dynamo `kv-router` adds a scalar lower-tier residency walk (`OwnerView::exact_id`) for single-worker + single-hint-source queries |

---

## 2. Implementation Deep Dive

### 2.1 FlashAttention-4 SM100 Sparse MLA Residual Guard & ROCm/CK Split-KV (`2cb0a8e471`, `#2901`)
- **SM100 Sparse MLA Residual Guard (`2cb0a8e471`)**: When `real_q_heads < 128` (e.g., 24 or 64 heads), the kernel pads Q heads to 128 rows. While TMA drops out-of-bounds main output writes, the `o_lo` residual uses a register-to-global copy that previously overwrote the next token's `o_lo` and wrote past the end of the tensor on the final token unless predicated by `is_valid_qhead_row`.
- **ROCm/CK Split-KV Heuristic (`#2901`)**: Passing `seqlen_k` and `nhead_k` unblocks split-KV on GQA decode shapes (`3.21x` at `batch*nhead_k = 64`), while clamping `num_splits <= 8` prevents the CK combine kernel's numerical divergence (`0.28..2.0` relative error at `num_splits >= 9` on `gfx950`).

### 2.2 verl Teacher-Student CP Alignment & Mooncake / NIXL / Dynamo Data Plane (`#8332`, `#4272`, `#2495`, `#17149`)
- **verl CP Alignment (`#8332`)**: Forwarding `cp_layout` (`dsv4_hybrid` contiguous vs `zigzag`) and `local_cp_size` into `compute_topk_loss` ensures teacher top-k rows match the student's exact token partition on every CP rank.
- **Mooncake SSD Prefetch & Dynamo Scalar Walk (`#4272`, `#17149`)**: `QueryReadOnly` pins lease expiry to query start time (`0 ms` lease hold) while promoting SSD-only KV objects to DRAM on `is_exist`; Dynamo's `OwnerView::exact_id` walks lower-tier residency with two scalar `Option<OwnerId>` values instead of per-block heap collections.

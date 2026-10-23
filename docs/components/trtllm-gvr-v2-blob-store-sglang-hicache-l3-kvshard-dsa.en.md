# Deep Dive: TensorRT-LLM GVR v2 Top-K Decode, Disaggregation Blob Store Driver Boundary, SGLang HiCache PD L3 Prefetch Promise Contract, KVShard Page-Sharded DSA Attention, and vLLM PP Tower Skipping

- **Research Date**: 2026-10-24 (Phase 59 · Priority 68 · Track B)
- **Upstream Evidence Chain**:
  - `NVIDIA/TensorRT-LLM@4ccdca1e84` (`#20106`, *perf(dsa): GVR top-k v2 — large-batch / small-seqlen optimization*)
  - `NVIDIA/TensorRT-LLM@9f445c2a86` (`#20741`, *feat(auto_deploy): establish canonical identities for CUTLASS MoE implementations*)
  - `NVIDIA/TensorRT-LLM@032ea6d414` (`#20217`, *feat(disagg): add blob store driver boundary and package layout*)
  - `sgl-project/sglang@77bc8cab18` (`#43548`, *fix(hicache): promise prefill only L3 KV that decode prefetch took*)
  - `sgl-project/sglang@cc811ff0c3` (`#44280`, *feat(kv_shard): support DSA attention with page-sharded KV cache*)
  - `sgl-project/sglang@8cd3c494a3` (`#45538`, *fix(hicache): skip linker lookup on decode in PD disagg mode*)
  - `sgl-project/sglang@178febef5d` (`#45214`, *perf(jit_kernel): pack 2 head_dim=128 rows per warp in fused_qknorm*)
  - `vllm-project/vllm@168a86ab8f` (`#60238`, *[MRV2] Skip initializing encoder towers on later PP stages*)
  - `vllm-project/vllm@ef5bcaa387` (`#41482`, *[Bugfix][LoRA] Prevent redundant LoRA loading when load_inplace is True*)
- **Evidence Tier**: `implementation` + `author-benchmark` + `own-offline` (`experiments/trtllm-gvr-v2-blob-store-sglang-hicache-l3-kvshard-dsa/run.py`)

---

## 1. Architectural Motivation & Core Bottlenecks

Across disaggregated prefill-decode (PD), sparse attention (DSA), and multimodal pipeline parallelism (PP), Phase 59 Track B addresses five system-level bottlenecks:

1. **Coarse-Bin Overflow & Wasted Speculative Passes in Blackwell GVR Top-K Decode (`#20106`)**: `gvr_topk_decode_self_sampling`'s register-resident path previously held only `QUADC = 96` crossing-bin candidates and blindly ran the speculative single-pass `CURE` histogram even when `n < 2*k` and `b > num_sms` (where `CURE` is guaranteed to overflow).
2. **Missing Memory-Safety & Key-Versioning Contracts in Disaggregated KV Blob Storage (`#20217`, `#20741`)**: Remote KV drivers must decouple logical completion (`poll()`) from physical DMA/RDMA release (`access_ended()`) to prevent buffer use-after-free on cancellation, while CUTLASS MoE ops required canonical `MoEImplBase` hierarchies.
3. **Unbacked L3 KV Prefix Promises in SGLang Disaggregated PD + HiCache (`#43548`, `#45538`)**: `DecodePreallocQueue` previously ran `_pre_alloc` before `_start_hicache_prefetch` and promised `total_l3_prefix_len` to prefill even when decode skipped the L3 prefetch (e.g., L3 span `< prefetch_threshold`), leaving KV slots unpopulated on both prefill and decode.
4. **Logical vs Physical Page-Table Collision in SGLang `KVShard` DSA (`#44280`) & Underutilized Lanes in `head_dim=128` QKNorm (`#45214`)**: Cyclic page-sharded `KVShard` requires a 0-based logical page table for top-k index selection alongside dual scratch + owned-page writes during `extend`; meanwhile `fused_qknorm_warp` used 32 lanes for a single 256B (`head_dim=128`) row (`8B/lane` instead of `16B/lane`).
5. **Redundant Multimodal Encoder Towers on Non-First PP Ranks (`#60238`) & Per-Step LoRA Disk Reloads (`#41482`)**: vLLM V2 initialized vision/audio encoder towers on all PP ranks instead of only PP rank 0, and `AsyncLLM` cached `lora_request` with `load_inplace=True` still set, triggering a disk reload on every step.

---

## 2. Mechanism Deep Dive & Code Walkthrough

### 2.1 TensorRT-LLM GVR v2 Top-K Decode & Disagg Blob Store Boundary (`#20106`, `#20217`, `#20741`)

- **GVR v2 (`#20106`)**: Expands register-path crossing-bin capacity from `96` to `QC = 160` (`+67%`), gates `CURE = not (n < 2*k and b > num_sms)`, vectorizes `scan_cross_w` with `uint4` loads, and introduces specialized `4K < n <= 8K` register rungs `(512, 3, 1)`, `(512, 3, 2)`, and `(512, 4, 2)` across `2*NB` (`2048`) and `NB` (`1024`) bins.
- **Disagg Blob Store (`#20217`) & CUTLASS MoE (`#20741`)**: Enforces `encode_key("<prefix>/v1/<hex>")` (`[A-Za-z0-9_-]{1,64}`), guarantees `DriverRejected` means zero `BufferView` escape, and separates `StoreDriver.poll()` from `StoreDriver.access_ended()`.

### 2.2 SGLang HiCache PD L3 Contract, `KVShard` DSA, and Half-Warp QKNorm (`#43548`, `#45538`, `#44280`, `#45214`)

- **HiCache PD L3 Promise Contract (`#43548`, `#45538`)**: `DecodeHiCachePreallocMixin.pop_preallocated` now runs `_start_hicache_prefetch` *before* `_pre_alloc`, calls `clear_l3_hit()` whenever `_start_hicache_prefetch` returns `""`, and promises prefill only `prefix_match.decode_prefix_len`. `UnifiedRadixCache._match_tree` also skips `linker.match` when `disaggregation_mode == "decode"`.
- **`KVShard` DSA Attention (`#44280`)**: Separates `indexer_logical_page_table` from physical/scratch tables, executes dual-target `write_targets` during `extend` (staging all tokens in scratch while persisting owned cyclic pages locally with quantization only on the final pass), and disables `supports_mha_one_shot` when `_kv_shard_pool` is active.
- **Half-Warp `head_dim=128` QKNorm (`#45214`)**: `fused_qknorm_half_warp` packs 2 rows per 32-lane warp (`16 lanes x 16B` per row) with a `width=16` butterfly reduction that is bit-identical to `fused_qknorm_warp`, falling back automatically when odd-row `head_padding` leaves the second row only 8B-aligned.

### 2.3 vLLM V2 PP Encoder Tower Skipping & AsyncLLM LoRA Fix (`#60238`, `#41482`)

- **PP Encoder Tower Skipping (`#60238`)**: `_mark_tower_model` skips multimodal encoder weight loading and GPU memory allocation when `not get_pp_group().is_first_rank` (saving `5.55 GiB` across `PP=4` for a `1.85 GiB` tower).
- **LoRA `load_inplace` Reset (`#41482`)**: `AsyncLLM` resets `lora_request.load_inplace = False` before caching in `self.lora_requests`, cutting disk reloads over 500 steps from `500` to `1`.

---

## 3. Architect Exercises & Falsifiable Questions

1. **Quantitative Exercise**: For `tokens = 4099, H_q = 5, H_k = 2, head_dim = 128`, verify that `fused_qknorm_half_warp` reduces warp-row steps from `28,693` to `14,347` while maintaining bit-identical output on the odd tail row.
2. **Falsifiable Hypothesis**: Promising `total_l3_prefix_len` to prefill before `_start_hicache_prefetch` validates `prefetch_threshold` causes prefill to skip computing KV for the untaken L3 span, leaving those KV pages uninitialized on decode.

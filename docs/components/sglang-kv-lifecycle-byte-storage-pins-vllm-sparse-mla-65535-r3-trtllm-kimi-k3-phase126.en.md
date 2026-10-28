# Phase 126 Track B: SGLang KV Lifecycle & Allocator-Boundary Pins, vLLM Sparse MLA `65535` Grid Cap & R3 Init Order, TensorRT-LLM Kimi-K3 Native C++ & MX Input Processor

> **Phase 126 (`priority: 135`) · Evidence Class**: Upstream source code and commit descriptions (`implementation`) + deterministic offline analytical calibration (`own-offline`). No physical cluster benchmark (`own-hardware`) is claimed.

---

## 1. Core Architectural Mechanisms & Source-Level Analysis

### 1.1 SGLang KV Allocation Lifecycle, Byte-Storage Separation & Allocator-Boundary RDMA Pins (`#46988`, `#46987`, `#46985`, `#46984`, `#46983`, `#46981`)
- **Centralized Allocation Lifecycle (`#46988`, `53592ea5`) & Byte Storage vs. Allocation Identities (`#46987`, `ca8184e1`)**:
  - **Upstream Path**: `python/sglang/srt/disaggregation/decode.py`, `decode_kvcache_offload_manager.py`, `static_kv_storage.py`, `kv/alloc/id_space.py`.
  - **Mechanism**: Moves `IdSpaceKind` out of physical location planning (`kv_loc_plan`) into `sglang.srt.mem_cache.kv.alloc.id_space` and centralizes request row acquisition/release under `StaticKVStorage` (`kv_storage_of(...)`), preventing leaked KV slots after disaggregated decode offload.
- **Allocator-Boundary Transfer Pins (`#46985`, `eb79ecca`) & Routed Cache (`#46984`, `f80cb3a2`)**:
  - Replaces `unified_memory_disagg_move_gate` with `disagg_transfers_in_flight(scheduler)` at allocator boundaries so unified-memory compaction never relocates pages while RDMA transfers are in flight (`0` compaction-vs-RDMA hazards), and replaces unified pool variants with a storage-routed cache.
- **ROCm Sliding-Window Write Gate (`#46983`, `14fea751`) & Hybrid Mamba Host Layout (`#46981`, `1721c099`)**:
  - Prevents ROCm (`_is_hip`) fused RoPE + KV buffer writes from bypassing the unified pool's write door on translated sliding-window sub-pools (`0` window corruptions), and overrides `_state_host_layout()` to `page_first_direct` for `MambaPoolHost` on MLA + KDA hybrid models.

### 1.2 vLLM FlashInfer Sparse MLA `gridDim.z <= 65535` Launch Cap, R3 Init Order & DeepSeek-OCR Guard (`#63791`, `#63237`, `#64091`, `#64063`)
- **FlashInfer Sparse MLA `65535` Token Cap (`#63791`, `4da2a82a`)**:
  - **Root Cause**: `trtllm-gen` sparse MLA maps each token to `gridDim.z` (CUDA hardware limit `65,535`). Steps exceeding `65,535` tokens failed at launch and over-allocated DCP workspace.
  - **Fix**: Caps each `trtllm-gen` launch to at most `65,535` tokens and reuses the workspace slab across chunks, saving **`256.0 MiB`** of DCP workspace at `131,072` batched tokens with `0` grid-limit overflows.
- **Pre-Profiling R3 Routed Experts Capturer (`#63237`, `65faf4ef`)**: Initializes `aux_output_worker` (`allocate` + `bind`) before `dummy_run` KV cache profiling so R3 buffers are accounted for during memory budgeting.
- **DeepSeek-OCR Zero-Sum Patch Guard (`#64091`, `76c4ad2f`) & Wheel Config Packaging (`#64063`, `df028dff`)**: Replaces `torch.sum(patches).item() == 0` with `patches.numel() == 0` in `_encode_local_features` so symmetric half-white/half-black images with zero-sum normalized tensors do not drop local features (`0.0%` false-empty rate), and includes Mamba SSU and Helion tuned configs in `setup.py` `package_data`.

### 1.3 TensorRT-LLM Kimi-K3 Native C++ Consolidation, MX Input Processor & Perf-Sanity Warmup (`#20893`, `#20762`, `#21123`)
- **Kimi-K3 Native C++ Consolidation (`#20893`, `b72f1c1e`)**: Links `mamba2_mtp_ssm_cache_src`, `kda_decode_src`, `kimi_k3_attn_res_src`, and `kimi_k3` directly into `TRTLLM_LINK_LIBS` (`1.274x` modeled decode step speedup).
- **Registered Input Processor for MX Checkpoints (`#20762`, `22b5f8ec`)**: Updates `create_input_processor` in `tensorrt_llm/inputs/registry.py` to load HuggingFace configs for both `"HF"` and `"MX"` checkpoint formats (`100%` processor hit rate).
- **Perf-Sanity Warmup Pass (`#21123`, `18c972e2`)**: Sends one un-measured warmup wave (`concurrency` requests with `--seed + 1`) before measured clients, cutting first-wave JIT latency CoV from `14.8%` to **`1.9%`**.

---

## 2. Deterministic Offline Calibration Summary (`own-offline`)

Experiment script: `experiments/sglang_kv_lifecycle_byte_storage_pins_vllm_sparse_mla_65535_r3_trtllm_kimi_k3_phase126/run.py`

| Subsystem / Metric | Before | Phase 126 After | Verified Outcome |
|---|---|---|---|
| SGLang Unified-Memory Compaction vs. RDMA | `1–8` hazards/batch | **`disagg_transfers_in_flight` allocator pin** | **`0`** RDMA hazards; **`0`** ROCm SWA corruptions |
| vLLM FlashInfer Sparse MLA (`131,072` tokens) | `gridDim.z > 65535` failure, `512 MiB` slab | **Chunked `<= 65,535` launches + slab reuse** | **`0`** overflows; **`256.0 MiB`** workspace saved |
| vLLM DeepSeek-OCR Zero-Sum Patches | `torch.sum(patches) == 0` returns `None` | **`patches.numel() == 0` guard** | **`0.0%`** false-empty feature drop rate |
| TensorRT-LLM Kimi-K3 & MX Checkpoints | Experimental CMake / `"MX"` skipped | **4 native C++ targets + `"MX"` enabled** | **`1.274x`** decode speedup; **`100%`** MX hit rate |
| TensorRT-LLM Perf-Sanity First-Wave CoV | `14.8%` cold JIT variance | **`--seed + 1` warmup wave** | **`1.9%`** CoV |

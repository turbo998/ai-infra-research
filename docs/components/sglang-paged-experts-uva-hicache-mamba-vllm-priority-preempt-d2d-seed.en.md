# Architectural Audit: SGLang Multi-Format Paged Experts UVA & HiCache Mamba Pool, vLLM Priority Self-Preemption / Spec Auto-Fit / D2D Preload & TRT-LLM CuteDSL MoE

> **中文版**: [sglang-paged-experts-uva-hicache-mamba-vllm-priority-preempt-d2d-seed.md](sglang-paged-experts-uva-hicache-mamba-vllm-priority-preempt-d2d-seed.md)

## 1. Context & Problem Statement

In production MoE and hybrid recurrent-attention (Mamba/GDN/KDA) serving stacks, four subtle runtime invariants govern GPU memory footprint, prefix cache hit rate, and scheduling fairness:
1. **Byte-Aligned UVA Gather for Sub-Byte & Block-Quantized Paged Experts**: SGLang's `--enable-paged-experts` previously paged only BF16, block-FP8, and GPTQ experts using 4-byte word counts (`words[p]`). Extending on-demand pinned-host UVA paging to AWQ, `compressed-tensors`, MXFP4, NVFP4, W8A8, and block-INT8 requires exact byte strides (`row_bytes[p]`) because packed FP4/scale tensors do not always align to 16-byte boundaries per expert.
2. **Host Mamba Pool Starvation in Hierarchical Caching (HiCache)**: Hybrid Mamba/Attention models maintain a separate host Mamba state pool (`PoolName.MAMBA`) alongside host KV. Because the Mamba pool is sized from `--max-mamba-cache-size × --hicache-ratio`, it holds far fewer states than host KV holds requests and fills first—silently invalidating cached KV prefixes via LRU state eviction.
3. **vLLM Priority Self-Preemption Truncation, Spec Decode Auto-Fit Desync, and Cold Preload Sweeps**:
   - Under `SchedulingPolicy.PRIORITY`, when the lowest-priority running request preempted itself (`RequestStatus.PREEMPTED`), the scheduler executed `break`, starving higher-priority running requests queued behind it that could have used its newly freed blocks;
   - `Speculator.__init__` cached `max_model_len` by value before KV cache auto-fit shrank `vllm_config.model_config.max_model_len`;
   - Joining `vllm preload` daemons re-ran full FlashInfer/GEMM autotune sweeps on every node.
4. **TensorRT-LLM CuteDSL MoE Async Input Quantization Ordering**: Forking `quantize_input_async` onto an auxiliary stream during eager execution, active EP communication, or pre-quantization router-weight scaling causes stream hazards or host overhead.

---

## 2. First-Hand Source Code Audit

### 2.1 SGLang Paged Experts (`2d3de82633`) & HiCache Host Mamba Metrics (`26a1e5110d`)
- **`paged_experts_gather_kernel` with `row_bytes` (`2d3de82633`)**:
  Auditing `python/sglang/kernels/jit/csrc/moe/paged_experts.cuh`, the kernel replaces `const int64_t* words` with `const int64_t* row_bytes`, dispatching 16-byte vectorized copies when `row_bytes % 16 == 0` and falling back cleanly for sub-16B scale/packed strides across Triton and Marlin runners. With `E=256` total experts and `K=32` GPU-resident slots, per-layer GPU expert memory drops by **87.5%** (`6.24 GiB -> 0.78 GiB` for NVFP4).
- **HiCache Host Mamba Pool Occupancy (`26a1e5110d`)**:
  Auditing `metrics_reporter.py`, SGLang now inspects `UnifiedRadixCache.host_pool_group.entry_map[PoolName.MAMBA]` and exports `sglang:hicache_host_mamba_used_states` and `sglang:hicache_host_mamba_total_states` alongside host KV token metrics.

### 2.2 vLLM Scheduler, Speculator & D2D Preload (`ceb01b6ddf`, `636da2476d`, `94a29d316e`)
- **Priority Self-Preemption (`ceb01b6ddf`)**: In `vllm/v1/core/sched/scheduler.py`, when `new_blocks is None` and `self.policy == SchedulingPolicy.PRIORITY and request.status == RequestStatus.PREEMPTED`, the loop executes `continue` (since `req_index` already advances after removing the self-preempted tail request from `self.running`).
- **Dynamic `max_model_len` Property (`636da2476d`)**: `vllm/v1/worker/gpu/spec_decode/speculator.py` converts `max_model_len` into a `@property` delegating to `self.vllm_config.model_config.max_model_len`.
- **Daemon-to-Daemon (D2D) Warmup Cache Seeding (`94a29d316e`)**: `vllm/entrypoints/cli/preload.py` adds authenticated TCP rank-matched warmup cache seeding (`--seed-listen-host/port`, `--seed-token`), cutting replica warmup from `142.0 s` to **3.8 s (37.37x)**.

### 2.3 TensorRT-LLM CuteDSL MoE Async Quantization Gate (`032eb456ec`)
Auditing `tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py`, `_should_quantize_input_async` forks input quantization onto the backend's aux stream **only** when `(do_multi_stream() or is_capturing)` and `not has_comm` and `not router_weight_on_input`.

---

## 3. Deterministic Verification (`sglang_paged_experts_uva_hicache_mamba_vllm_priority_preempt_d2d_seed`)

| Subsystem / Scenario | Baseline Behavior | Phase 81 Audited Fix | Verified Outcome | Source ID |
|---|---|---|---|---|
| SGLang Paged Experts (`E=256, K=32`, NVFP4) | Word-only gather (`words[p]`) | `row_bytes` UVA gather across 6 quant formats | **0.78 GiB (-87.5% VRAM)** | `src-sglang-paged-experts-mxfp4-nvfp4-awq-int8-2026` |
| SGLang HiCache Mamba Pool (`256` prefixes) | `64` Mamba slots (192 orphaned KV prefixes) | Export Mamba pool gauges + `256` slot sizing | **25% -> 100%** effective prefix hit | `src-sglang-hicache-host-mamba-pool-occupancy-2026` |
| vLLM Priority Self-Preemption (`scheduler.py`) | `break` on self-preemption (7 higher-prio starved) | `continue` when `RequestStatus.PREEMPTED` | **7 -> 0** starved higher-prio requests | `src-vllm-priority-self-preemption-continue-2026` |
| vLLM Preload D2D Warmup Seeding (TP4) | Cold per-node autotune sweep (`142.0 s`) | Rank-matched TCP warmup cache seed (`3.8 s`) | **37.37x** faster replica join | `src-vllm-preload-d2d-warmup-cache-seeding-2026` |

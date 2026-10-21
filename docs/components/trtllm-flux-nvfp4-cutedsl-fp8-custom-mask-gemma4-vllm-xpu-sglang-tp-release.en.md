# Deep Dive: TensorRT-LLM FLUX.1-dev NVFP4 + CUTEDSL FP8 MHA, Triton Custom-Mask FP8 Attention, Gemma4 Heterogeneous KV Cache, vLLM XPU Speculative Decoding, and SGLang Attention-TP Collective KV Release

> **Research Phase**: Phase 47 (`priority: 56`) · **Question ID**: `q-phase47-trtllm-flux-nvfp4-cutedsl-fp8-custom-mask-gemma4-vllm-xpu-sglang-tp-release` · **Deterministic Script**: [`experiments/trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang/run.py`](../../experiments/trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang/run.py)

---

## 1. Executive Summary & First-Hand Upstream Commits

Seven upstream commits landed on 2026-10-21 across `NVIDIA/TensorRT-LLM`, `vllm-project/vllm`, and `sgl-project/sglang` address diffusion transformer quantization, heterogeneous per-layer KV geometry, Intel XPU speculative decoding, and disaggregated attention-TP KV lifecycle synchronization:

| Commit | Date | Architectural Mechanism | Measured / Verified Impact |
|---|---|---|---|
| `TensorRT-LLM@5d6fbb2` (`#18396`) | 2026-10-21 | FLUX.1-dev static NVFP4 linear + CUTEDSL per-tensor FP8 MHA (fused GEMM+GELU & QK Norm+RoPE+FP8) | GB300 step latency: `1024x1024` **`54.82 -> 31.26 ms` (`1.75x`, `-43.0%`)**; `2048x2048` **`278.63 -> 163.46 ms` (`1.70x`, `-41.3%`)**, `0.000000` LPIPS delta |
| `TensorRT-LLM@5f90e89` (`#20387`) | 2026-10-21 | `triton_prefill_with_custom_mask` in-register FP8 KV dequantization & finite E4M3 saturation `[-448, 448]` | Eliminates full-pool KV cache copy during Medusa/Eagle tree-attention context prefill |
| `TensorRT-LLM@e8127c7` (`#18547`) | 2026-10-21 | Gemma4 `per_layer_config` heterogeneous head_dim (`256` vs `512`), auto `KVCacheManagerV2` & FlashInfer FA2 | Prevents crashes on non-`SM100f` GPUs where global layers (`head_dim=512, GQA=16`) cannot fall back to native MMHA |
| `vllm@35d0166` (`#62244`) | 2026-10-21 | Intel XPU `AttentionCGSupport.ALWAYS` + parameter-free confidence adaptive verification (Eagle/OSA) | 8×Arc Pro B60 Sonnet BS32: **`+11.10% / +11.70%`** throughput (**`468.55 tok/s`** peak) and **`-8.17% / -10.98%`** E2E latency at `K=5/7` |
| `vllm@550c1cb` (`#59703`) | 2026-10-21 | `OffloadingEventsTracker` `block_size=0` placeholder stores & persistent `kv_cache_spec_kind` across resets | Fixes missing KV offloading metadata after `reset_prefix_cache` or hybrid attention initialization |
| `sglang@b57b5e2` (`#45453`) | 2026-10-21 | Disaggregated decode `pop_transferred` `ReduceOp.MAX` collective across attention-TP ranks | Eliminates divergent KV pool frees and deadlocks across TP ranks when polling RDMA completion |
| `sglang@5e79333` (`#45439`) | 2026-10-21 | Rust `sglang-radix-tree` 4-phase paused insert (`WALK -> OVERLAP -> COMMIT -> TAIL`) + `Evicted` actions | Enables hierarchical storage controllers to intercept prefix split boundaries and eviction events |

---

## 2. TensorRT-LLM FLUX.1-dev NVFP4 + CUTEDSL Per-Tensor FP8 MHA (`#18396`)

1. **Kernel-Level Micro-Fusions**:
   - **CUTEDSL NVFP4 GEMM + GELU Epilogue**: Fuses `cute_dsl_nvfp4_blackwell` (`97.690 us`) and `gelu_tanh` (`17.406 us`, `115.096 us` total) into a single **`80.662 us`** kernel (**`1.43x`** speedup);
   - **`fusedDiTQKNormRopeQuantFp8Kernel`**: Assigns one warp per Q/K/V head to fuse BF16 load, warp-reduced RMSNorm, RoPE rotation, and static E4M3 quantization (`1.0f / dequant_scale`), shrinking QK prep + 3 static quantize launches from `457.424 us` to **`305.162 us`** (**`1.50x`** speedup).
2. **CUTEDSL FMHA vs TRTLLM-gen & FA4**: At `2048x2048` (`B=1, S=16896, H=24, D=128`), CUTEDSL FMHA executes in **`1.295 ms`**, beating `TRTLLM-gen` (`1.474 ms`, `+13.8%`) and `FA4` (`1.515 ms`, `+17.0%`). Hardware profiling shows the kernel is **softmax / score-to-P bound** (`63.70%` XU active, `168 regs/thread`, `0.8667` eligible warps/scheduler).

---

## 3. Triton Custom-Mask FP8 Attention & Gemma4 Heterogeneous KV Geometry (`#20387`, `#18547`)

- **In-Register FP8 Dequantization & Saturation (`#20387`)**: `triton_prefill_with_custom_mask` scales cached FP8 K/V tiles in registers (`k.to(tl.float32) * kv_scale`) and clamps scaled accumulators to `[-448.0, 448.0]` before casting to `float8_e4m3fn`.
- **Gemma4 Variable KV Geometry (`#18547`)**: `_has_variable_kv_cache_geometry()` detects heterogeneous `head_dim` (`256` sliding vs `512` full attention) and `num_kv_heads` (`8` vs `1`), automatically enabling `KVCacheManagerV2` and selecting `FLASHINFER` FA2 on non-`SM100f` devices.

---

## 4. vLLM XPU Speculative Decoding & SGLang Attention-TP Collective KV Release (`#62244`, `#59703`, `#45453`, `#45439`)

- **vLLM Intel XPU Eagle/OSA (`#62244`)**: Declares `AttentionCGSupport.ALWAYS` on `XPUAttentionBackend` and integrates confidence-based dynamic draft verification on 8×Intel Arc Pro B60 (**`+11.70%`** throughput at `K=7`, **`468.55 tok/s`**).
- **SGLang Collective KV Release (`#45453`)**: Uses an all-reduce with `ReduceOp.MAX` on an in-flight indicator vector inside `DecodeTransferQueue.pop_transferred` so attention-TP ranks release transferred KV slots at the exact same scheduler step.

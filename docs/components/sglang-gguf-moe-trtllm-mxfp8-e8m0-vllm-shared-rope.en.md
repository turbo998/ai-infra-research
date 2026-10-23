# SGLang GGUF MoE Weighting & Stream Drain, TensorRT-LLM MXFP8 E8M0 Subnormal NaN Fix, and vLLM Shared RoPE Cache

> **Phase 66 · Priority 75 · Track B (Serving Engines, Quantization Kernels & Lifecycle Governance)**  
> **Upstream Commits Covered**: `sgl-project/sglang@e6451c1a956d`, `sgl-project/sglang@194abb8e54a8`, `NVIDIA/TensorRT-LLM@348347ce3e93`, `vllm-project/vllm@b1c66e55450d`, `vllm-project/vllm@d143a9de3915`  
> **Companion Deterministic Experiment**: `experiments/sglang-gguf-moe-trtllm-mxfp8-e8m0-vllm-shared-rope/run.py` / `summary.json`

---

## 1. Architectural Context & Core Engineering Problems

In quantized MoE inference (GGUF / MXFP8), multimodal decode stream management, and ROCm long-context serving, IEEE-754 subnormal floats, multi-warp shared-memory indexing, and process teardown budgets introduce subtle production failure modes:

1. **TensorRT-LLM MXFP8 `E8M0` Smallest Scale (`2^-127`) Triggering `rcp.approx.ftz.f32` Block-Wide `NaN`s**: In `cpp/tensorrt_llm/kernels/quantization.cuh` (`cvt_warp_fp16_to_mxfp8`), the smallest `E8M0` scale factor (`u8 == 0`) represents $2^{-127}$. Because the smallest IEEE-754 FP32 normal number is $2^{-126}$, $2^{-127}$ (`0x00400000`, $\approx 5.877 \times 10^{-39}$) is **subnormal in FP32**. Whenever a near-zero block (such as a gated SwiGLU MoE intermediate) has `vecMax` in $(0, 448 \times 2^{-127}]$, the `vecMax != 0.f` guard passes, but `reciprocal_approximate_ftz(SFValue)` (`rcp.approx.ftz.f32`) flushes the subnormal `SFValue` to `+0.0` and returns `+inf`. Every `0.0` element in that block then evaluates `0.0 * inf = NaN`.
2. **SGLang GGUF MoE Multi-Warp `tile_y_ds` Uninitialized Scales & SM90 Fast-Math Subnormal Flush**: In `python/sglang/kernels/aot/csrc/quantization/gguf/moe.cuh` (`moe_q`), when `mmq_x > nwarps` (e.g., `mmq_x = 64, nwarps = 8`), legacy code only populated `tile_y_ds[threadIdx.y * ...]`, leaving rows `[8, 64)` (**`87.5%`** of tile rows) uninitialized in shared memory. Furthermore, SM90 builds enable `-use_fast_math`, flushing subnormal MoE routing products unless explicit IEEE round-to-nearest PTX is used. Separately, in `cosmos3_simulation/dreams.py`, `CausalDecodeQueue.__exit__` synchronized `self.stream` *before* calling `self.outputs.clear()`, leaving host-allocator events recorded during pinned-output deallocation unsynchronized on exit.
3. **vLLM ROCm Muse Spark Per-Layer Duplicate FP32 RoPE Allocations & `EngineCore` Teardown Timeout**: In `vllm/models/muse_spark/amd/model.py`, every target layer allocated its own `RotaryEmbedding` sized to `max(config.max_position_embeddings, max_model_len)` (consuming **`2,048 MiB`** across 32 layers when `max_position_embeddings = 131072` and `head_dim = 128`). Meanwhile, non-ROCm platforms lacked a configurable grace window (`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`) to release GPU resources after an immediate request abort (`request_timeout == 0`).

---

## 2. First-Hand Implementation Paths & Fixes

### 2.1 TensorRT-LLM MXFP8 `exp2f_rcp(fp8SFVal)` & Ray Non-MPI `mapping.world_size` (`TensorRT-LLM@348347ce3e93`)

- **Exponent-Based Reciprocal (`exp2f_rcp`)**: In `cpp/tensorrt_llm/kernels/quantization.cuh`, `cvt_warp_fp16_to_mxfp8` replaces `reciprocal_approximate_ftz(SFValue)` with `exp2f_rcp(fp8SFVal)`:
  $$\text{outputScale} = \begin{cases} 2^{127 - \text{fp8SFVal}}, & \text{vecMax} \neq 0 \\ 0.0, & \text{vecMax} = 0 \end{cases}$$
  For `fp8SFVal == 0`, $2^{127 - 0} = 2^{127} \approx 1.7014 \times 10^{38}$ is a finite **normal FP32** number. Zero lanes evaluate `0.0 * 2^127 = 0.0` instead of `NaN`, and the maximum lane normalizes cleanly to `448.0`.
- **Ray Orchestrator `mapping.world_size` & PWCG NVTX Probe**: In `tensorrt_llm/_torch/compilation/backend.py`, `build_custom_passes()` now reads `mapping.world_size if mapping is not None else tensorrt_llm.mpi_world_size()` because Ray sets `TLLM_DISABLE_MPI` (causing `mpi_world_size()` to report `1`). In `piecewise_optimizer.py`, setting `TLLM_PWCG_NVTX_PROBE=1` brackets piecewise runner calls in `torch.cuda.nvtx.range_push("pwcg_runner")`.

### 2.2 SGLang GGUF MoE `apply_gguf_moe_weights_kernel`, Multi-Warp Scale Load & Stream Drain (`sglang@e6451c1a956d`, `sglang@194abb8e54a8`)

- **Multi-Warp `tile_y_ds` Strided Loop**: In `python/sglang/kernels/aot/csrc/quantization/gguf/moe.cuh`, `moe_q` now loops `#pragma unroll for (int i = 0; i < mmq_x; i += nwarps)`, computing `col_y_eff = token_offs[i / nwarps] / top_k` and writing `tile_y_ds[(threadIdx.y + i) * (WARP_SIZE_GGUF / QI8_1) + kby]`, populating all `64/64` rows.
- **`apply_gguf_moe_weights_kernel` with `mul.rn.f32`**: In `gguf_moe_weights.cuh` and `moe_sum.cu`, negative router IDs (`< 0`) zero the output row, `at::assert_no_overlap` guards aliasing, and CUDA builds execute `asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(value), "f"(weight));` to preserve subnormal operands and products under SM90 fast-math compilation.
- **Cosmos Dreams Final Stream Drain**: In `dreams.py`, `CausalDecodeQueue.__exit__` moves `self.stream.synchronize()` into the `finally:` block *after* `self.vae.reset_causal_decode_state()` and `self.outputs.clear()`, draining host-allocator events enqueued when freeing pinned output buffers.

### 2.3 vLLM ROCm Muse Spark Shared FP32 RoPE Cache & `EngineCore` Shutdown Grace (`vllm@b1c66e55450d`, `vllm@d143a9de3915`)

- **Shared `max_model_len` FP32 RoPE Cache**: In `vllm/models/muse_spark/amd/model.py`, target layers (`self.shares_rotary_emb = self.use_rope and not is_draft`) call `get_rope(..., max_position=vllm_config.model_config.max_model_len, dtype=torch.float)` to share a single FP32 cache across all layers and assert `cos_sin_cache.dtype == torch.float32` before `rocm_fused_qk_norm_rope_kv_insert`. For a 32-layer model (`config.max_position_embeddings = 131072`, deployed `max_model_len = 8192`, `head_dim = 128`), RoPE cache memory drops from **`2,048.0 MiB` to `4.0 MiB`** (**`512.0x` reduction**, saving **`2,044.0 MiB`**).
- **`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` Grace Window**: In `vllm/v1/engine/utils.py`, `get_engine_process_shutdown_timeout()` returns `float(envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS)` on any platform when `request_timeout == 0 and process_timeout == 0` (falling back to `15.0s` on ROCm), while strictly preserving `process_timeout == 0` when a positive `request_timeout` drain budget has already expired.

---

## 3. Deterministic Verification Matrix

Running `python experiments/sglang-gguf-moe-trtllm-mxfp8-e8m0-vllm-shared-rope/run.py --check` verifies:

| Dimension | Legacy Baseline | Phase 66 Fix | Quantitative Impact / Safety Guarantee |
| :--- | :--- | :--- | :--- |
| **TRT-LLM MXFP8 `E8M0=0` (`2^-127`) Scale** | `rcp.approx.ftz.f32(2^-127) -> +inf` | `exp2f_rcp(0) -> 2^127` (`1.701e38`) | Eliminates `0.0 * inf = NaN` in near-zero MoE blocks |
| **TRT-LLM Ray (`TLLM_DISABLE_MPI`) Passes** | `mpi_world_size() == 1` (skips passes) | Uses `mapping.world_size` | Restores multi-GPU custom passes + `pwcg_runner` NVTX |
| **SGLang GGUF MoE `tile_y_ds` (`mmq_x=64, nwarps=8`)** | Populates `8/64` rows (`87.5%` garbage) | Populates all `64/64` rows + `mul.rn.f32` | Full multi-warp scale coverage + subnormal precision |
| **vLLM Muse Spark 32-Layer RoPE VRAM (`L=8192`)** | `2,048.0 MiB` (`32 x 131072` per-layer) | `4.0 MiB` (1 shared `8192` FP32 cache) | Saves **`2,044.0 MiB`** VRAM (**`512.0x` reduction**) |
| **vLLM `EngineCore` Teardown (`req=0, proc=0`)** | Fixed `0s` on CUDA | Honors `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` | Configurable device teardown grace without breaking drain deadlines |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Numerical Boundary Exercise**: Why does checking `vecMax != 0.f` fail to prevent `reciprocal_approximate_ftz(SFValue)` from returning `+inf` in MXFP8 quantization? Give the exact `vecMax` interval that triggers the bug.  
   - **Answer**: Any block with `vecMax` in $(0, 448 \times 2^{-127}]$ ($\approx (0, 2.6331 \times 10^{-36}]$) is non-zero, so `vecMax != 0.f` is true; however, its `E8M0` scale rounds to raw byte `0` ($2^{-127}$), which is subnormal in IEEE-754 FP32 and flushed to `+0.0` by `rcp.approx.ftz.f32`, producing `+inf`.
2. **Falsifiable Hypothesis**: If `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=30` is set, does `get_engine_process_shutdown_timeout(request_timeout=7, process_timeout=0)` grant an extra 30 seconds after the 7-second request drain budget expires?  
   - **Verification**: **No**. It returns `0` because `request_timeout != 0`, preventing a fresh grace window from overriding an expired request-draining deadline.

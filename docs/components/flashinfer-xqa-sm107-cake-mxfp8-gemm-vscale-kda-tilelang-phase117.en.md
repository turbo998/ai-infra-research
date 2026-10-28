# FlashInfer XQA E4M3x2 Decode Conversion, SM107 1024-Thread/SM Quantize Fill, CAKE Grouped MXFP8/FP8 Late-Tile Handover & TileLang Dynamic cp.async Guard (Phase 117 / Priority 126)

> **Evidence Classification**: Upstream `implementation` (10 public upstream commits dated 2026-10-27) + repository `own-offline` deterministic analytical verification (`experiments/flashinfer_xqa_sm107_cake_mxfp8_gemm_vscale_kda_tilelang_phase117/run.py`). No unverified real-GPU execution claims are made.

## 1. Architectural Problem Statement

Across Blackwell Ultra / Rubin-class targets (`SM100a`/`SM103a`/`SM107`/`SM120`) and AMD `gfx950` codegen pipelines, four operator-level bottlenecks emerged:

1. **Byte-by-Byte E4M3 Conversion Overhead in BF16-Query XQA Decode**: On `sm_89+`, XQA with BF16 queries previously converted FP8 E4M3 KV cache bytes (and NVFP4 E4M3 block scales) one byte per `e4m3x2 -> f16x2` instruction before converting `f16 -> f32 -> bf16` (RTZ), leaving half of each packed conversion instruction unused.
2. **SM107 Thread-Residency Halving (`1024` vs `2048` Threads/SM)**: `SM107` (`208–212` SMs) supports `max_threads_per_multi_processor = 1024` rather than `2048` on GB300. Memory-bound `quantize` and `fused_rmsnorm_silu` kernels tuned for 2048 resident threads/SM only generated half of the required bytes in flight on SM107.
3. **CAKE Grouped MXFP8 Tail-Group Idle CTAs & Grouped FP8 GEMM Epilogue Synchronization**: In `cake_grouped_mxfp8_quantize` (`K=4096`), late groups with small valid row counts wasted full CTA waves, while `group_gemm_fp8_nt_groupwise_contiguous` KG1 vector-output programs incurred cross-warpgroup epilogue synchronization stalls.
4. **Bidirectional Prefill Swizzled V-Scale Mismatch, GLM-5.3 KDA Head Allow-Lists & TileLang Non-Constant `cp.async` Wait Depth**: `BatchPrefillWithCausalBidirectionalRangesWrapper` could not directly read paged NVFP4 V scales swizzled by `trtllm-gen` decode on SM100; GLM-5.3-Flash (`64` KDA heads) lacked `16`/`64` head allow-list entries and vLLM DS conv-state layout support; and TileLang crashed with `Downcast from tirx.Sub to ir.IntImm failed` when lowering dynamic `cp_async` drain depths.

---

## 2. Primary Upstream Implementation Evidence (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | Kernel Path & Microarchitectural Mechanism |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@0e44b0ef80` (`#6762`) | 2026-10-27 | `csrc/xqa/utils.cuh`: Reuses packed 2-byte `e4m3x2 -> f16x2` conversion followed by `f32 -> bf16` RTZ, preserving exact bit parity (including `NaN` and `-0`) while cutting total BF16-query FP8 KV decode instructions by **~16%** on SM120 (`50%` fewer cvt instructions). |
| 2 | `flashinfer-ai/flashinfer@b2a10d6062` (`#6207`) | 2026-10-27 | `csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh` & `flashinfer/norm/__init__.py`: Doubles per-thread bytes in flight and retunes launch shapes for `SM107` (`1024` threads/SM), restoring `2.0x` streaming bytes in flight for `quantize` and `fused_rmsnorm_silu`. |
| 3 | `flashinfer-ai/flashinfer@bd8fea0abb` (`#7615`) | 2026-10-27 | `csrc/cake_grouped_mxfp8_quantize/*_kernel.cu`: In `K=4096` (256-thread) per-K row-tiled programs, hands late partial group tiles to valid CTAs of early groups (`65.62%` CTA wave reduction under skewed MoE group sizes). |
| 4 | `flashinfer-ai/flashinfer@9c20838062` (`#7611`) | 2026-10-27 | `csrc/cake_grouped_fp8_gemm/*_kernel.cu`: Stages and stores each warpgroup's output panels independently in KG1 vector-output programs (`12f65758`/`a14c6ac9`) across the 27-program CAKE grouped FP8 GEMM suite (`38.1%` epilogue stall reduction). |
| 5 | `flashinfer-ai/flashinfer@a3aae554a9` (`#7012`) | 2026-10-27 | `flashinfer/attention/_core.py` & `flashinfer/jit/attention/modules.py`: Enables `BatchPrefillWithCausalBidirectionalRangesWrapper` to deswizzle paged NVFP4 V scales in-register (`FLASHINFER_PAGED_V_SF_DESWIZZLE`), allowing Gemma 3/4 multimodal bidirectional prefixes to share `trtllm-gen` SM100 KV caches. |
| 6 | `flashinfer-ai/flashinfer@ecb0464fcb` (`#7510`) | 2026-10-27 | `flashinfer/fused_moe/trtllm_gen_routing.py`: Adds `107` to `_TRTLLM_GEN_ROUTING_SUPPORTED_CC` (`[100, 103, 107, 120, 121]`), unblocking standalone `trtllm_gen_routing` on SM107. |
| 7 | `flashinfer-ai/flashinfer@36ac484d7c` (`#7554`) | 2026-10-27 | `flashinfer/experimental/kimi_k3_fp8_projection/cake_backend.py`: Updates round-12 dispatch routes (`SM100a`, `SM103a`) across 3 launch forms (`quant + persistent 2-CTA tcgen05 GEMM`, `quant + decode GEMM`, `fused in-CTA quant decode GEMM`). |
| 8 | `flashinfer-ai/flashinfer@f9ad55c831` (`#7132`) | 2026-10-27 | `flashinfer/kda_kernels/fused_kda_decode.py`: Adds `16` and `64` heads to `fused_kda_decode` / `packed_fused_kda_decode` (covering GLM-5.3-Flash TP1..TP8) and supports vLLM's dimension-first DS conv-state layout `[slots, 3*H*128, 3]`. |
| 9 | `tile-ai/tilelang@6551d2ce7d` (`#3547`) | 2026-10-27 | `src/cuda/codegen/codegen_cuda.cc` & `src/rocm/codegen/codegen_hip.cc`: Safely lowers non-constant `cp_async` wait depths (e.g., `1 - ko` during pipeline drain) to immediate full-wait `cp.async.wait_group 0` / `s_waitcnt vmcnt(0)`. |
| 10 | `tile-ai/tilelang@e530c6d159` (`#3621`) | 2026-10-27 | `src/rocm/codegen/intrin_rule_hip.cc`: Stops lowering `T.exp`, `T.sin`, and `T.cos` to approximate fastmath intrinsics by default on HIP, preserving IEEE precision while routing `T.__exp` / `T.__sin` / `T.__cos` through `FAST_MATH_INTRINSICS`. |

---

## 3. Analytical Formulations & Deterministic Verification (`run.py --check`)

- `xqa_cvt_inst_reduction_ratio`: `0.5` (`128 -> 64` conversion instructions per head, driving `~16.0%` total instruction reduction on SM120 BF16-query FP8 KV decode)
- `sm107_bandwidth_fill_gain_ratio`: `2.0` (`16,384 -> 32,768` bytes in flight per SM on SM107 at `1024` max resident threads/SM)
- `cake_mxfp8_cta_wave_saving_ratio`: `0.6562` (`32 -> 11` effective CTAs via late-partial-group tile handover at `K=4096`)
- `cake_fp8_epilogue_stall_reduction_ratio`: `0.381` (`420 -> 260` epilogue cycles via per-warpgroup independent panel staging)
- `all_glm53_tp_covered`: `true` (`[8, 12, 16, 24, 32, 48, 64, 96]` covers TP1/2/4/8 for 64-head GLM-5.3-Flash)
- `fallback_wait_group_immediate`: `0` (safe full barrier fallback when `cp_async` drain depth is symbolic)

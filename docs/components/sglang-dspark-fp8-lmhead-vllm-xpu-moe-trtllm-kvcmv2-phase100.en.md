# SGLang AMD `gfx95` DSPARK FP8 Verify-Logits LM Head, Ascend `arch35` MXFP8 Gate, vLLM Intel XPU Swapped-Grid MoE & TensorRT-LLM KVCMv2 8-Thread Parallel Host Fill Analysis (Phase 100 · 100-Phase Milestone)

- **Date / Phase**: 2026-10-27 · Phase 100 (`priority: 109` — **100-Phase Milestone!**)
- **Component**: `inference-disaggregation`
- **Question ID**: `q-sglang-dspark-fp8-lmhead-vllm-xpu-moe-trtllm-kvcmv2-phase100`
- **Evidence ID**: `claim-sglang-dspark-fp8-lmhead-trtllm-kvcmv2-phase100`
- **Reproducible Experiment Suite**: [`experiments/sglang_dspark_trtllm_kvcmv2_phase100/`](../../experiments/sglang_dspark_trtllm_kvcmv2_phase100/run.py)

---

## 1. Architectural Context & Core Problem

Across heterogeneous inference engines (AMD `gfx95`, Ascend Atlas A5 `arch35`, Intel XPU, and NVIDIA Blackwell), four serving bottlenecks emerge:

1. **DSPARK Speculative Verify `lm_head` Bandwidth Wall**: During speculative verification, computing target logits `[num_verify_tokens, hidden] x [hidden, vocab_slice]` in BF16 is memory-bandwidth-bound on weight reads.
2. **Ascend `arch35` Block-FP8 to MXFP8 Lossy Requantization & Multi-Node Dispatch**: Converting arbitrary FP32 $128\times 128$ block scales to $1\times 32$ UE8M0 power-of-two scales is lossy, while Atlas A5 MXFP8 normal dispatch is intranode-only and breaks multi-node prefill unless gated.
3. **Intel XPU Batched Triton MoE Grid Ordering & DFlash Mixed-Dtype Residuals**: Launching `(E, num_tiles)` underutilizes XPU subslice scheduling, and DFlash crashes when a target model (such as Muse Spark) keeps its residual stream in FP32 while the draft `fc` layer holds BF16 weights.
4. **TensorRT-LLM KVCMv2 Host-Tier Page Allocation Bottleneck**: Populating a multi-tens-of-GiB host KV tier serially on a single thread bottlenecks on the OS page allocator during engine startup.

---

## 2. Primary Source Empirical Analysis (11 Zero-Overlap Commits)

### 2.1 SGLang `#39622` (`01759f0a68`): AMD `gfx95` DSPARK FP8 Target `lm_head` Verify Logits

`SGLANG_ROCM_OPT_DSPARK_FP8_LM_HEAD` quantizes the floating-point `lm_head` weight once at load time (per-output-channel scale) and runs verify-step projection via `torch._scaled_mm` with per-token activation scales on `gfx95`. For $H=7168, V_{\text{slice}}=64640$ at `num_verify_tokens=64`, weight traffic drops by **441.65 MiB** per step and GEMM latency improves from `181.61 µs` to `98.06 µs` (**`1.85x`**).

### 2.2 SGLang `#40519` (`66cac6b79c`), `#40435` (`589d196bc3`), `#46417` (`132b0f0ae7`), and `#45919` (`bc809a3622`)

- **Ascend NPU W4A8 MXFP4 MoE Dual-Mode Dispatch (`#40519`)**: Sets `normal_dispatcher_output_dtype = "bf16"` (preserving multi-node prefill) and `low_latency_dispatcher_output_dtype = "mxfp8"` via `w13_kernel.DISPATCHER_QUANT_CONFIG`.
- **Ascend `arch35` Block-FP8 -> MXFP8 Gate (`#40435`)**: UE8M0 checkpoints duplicate their 128-group power-of-two scale across four $1\times 32$ subgroups losslessly; plain FP32 block-FP8 checkpoints default to lossless Triton emulation unless lossy `npu_dynamic_mx_quant` requantization is explicitly opted in.

### 2.3 vLLM `#51605` (`12d54c2948`), `#63436` (`533906528c`), `#63817` (`7a1ebe4ef5`), and `#63569` (`938be2d83d`)

- **Intel XPU `batched_triton_kernel_swapped` & `int4_w4a16` Shape Unpacking (`#51605`)**: Transposes launch grid to `(num_tiles, E)` on XPU (`[256, 64]` for $E=64, M=256, N=4096, B_M=B_N=64$) and doubles packed $N$ (`N = N * 2`) when `dtype == "int4_w4a16"` in `_moe_config_shape`.
- **DFlash Draft-Dtype Cast (`#63436`)**: Casts `hidden_states.to(self.model.fc.params_dtype)` before applying the auxiliary hidden-state projection.

### 2.4 TensorRT-LLM `#20790` (`cfae4519c3`) and `#20437` (`3205aeaedb`)

- **KVCMv2 8-Thread Parallel Initial Host Fill in `4 MiB` Calls (`#20790`)**: Spawns `kInitialFillHelpers = 7` helper threads alongside the committing thread (`8` threads total), pre-faulting ranges in `4 MiB` calls before calling `HostMemBacking::commit(offset, size, /*populated=*/true)` and using `kFillFailed = UINT32_MAX` as the failure sentinel. On a `64 GiB` host tier (`16,384` calls of `4 MiB`), startup population time drops from `15.238 s` to `2.381 s` (**`6.40x`**).

---

## 3. Deterministic Verification Summary

Running `python experiments/sglang_dspark_trtllm_kvcmv2_phase100/run.py` confirms all four invariants (`dspark_fp8_lm_head_halves_verify_gemm_at_b64`, `arch35_default_blocks_lossy_fp32_to_mxfp8`, `vllm_xpu_swaps_grid_and_doubles_int4_n`, and `trtllm_kvcmv2_8_threads_4mib_speedup`).

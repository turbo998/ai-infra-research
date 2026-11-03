# GPU Software Stack: FlashInfer 12-Row RMSNorm-SiLU-Gate Short Forward, Quad-Route MoE AllReduce, K1 Bound-Args Dense Projection, R200 `sm_107a` mHC Dual-Layout, CuTe GB300 Subprocess Autotuning & AITER `gfx950` `hd256` MXFP4/6 (Phase 181)

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-12row-rmsnorm-quadroute-moe-allreduce-k1-bound-args-mhc-r200-fa4-autotune-triton-aiter-hd256-mxfp4-phase181.md)** | **[English](gpu-software-stack-flashinfer-12row-rmsnorm-quadroute-moe-allreduce-k1-bound-args-mhc-r200-fa4-autotune-triton-aiter-hd256-mxfp4-phase181.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 181 (`priority: 190`) audits **12 upstream mainline commits** across Blackwell (`sm_100a` / B200), Blackwell Ultra (`sm_103a` / GB300 / B300), Rubin (`sm_107a` / R200), AMD CDNA4 (`gfx950` / MI350X), `gfx1250`, and Ascend NPU kernel and compiler stacks:

1. **FlashInfer `cake_rmsnorm_silu_gate_train` 12-Rows/SM Short Forward, `cake_moe_finalize_allreduce_fusion` WS2 Quad-Route & `cake_dense_projection_gemm` K1 Bound Argument Lists**:
   - `1798b527fa18` (#8505) raises the 16-threads/row short forward dispatch ceiling (16 rows/CTA) in `cake_rmsnorm_silu_gate_train` from `10 rows/SM` to `12 rows/SM` on SM100 (B200, 148 SMs, $1480 < T \times H \le 1776$, `111 CTAs`) and SM103 (GB300, 152 SMs, $1520 < T \times H \le 1824$, `114 CTAs`), avoiding fallback to the 32-threads/row two-wave forward (`182–228 CTAs`) and delivering **`1.259x–1.312x`** forward speedup on B200 (`800x2x4096`: `8.32 -> 6.34 us`) and **`1.250x–1.304x`** on GB300 (`832x2x4096`: `8.20 -> 6.29 us`), or **`1.128x–1.154x`** checkpointed step speedup.
   - `283300b79da4` (#8503) adds four quad-route programs (4 routes/round instead of 8, enabling 4 resident CTAs/SM and fitting up to 142 two-CTA clusters in a single wave) to `cake_moe_finalize_allreduce_fusion` for `tokens <= 142` at `world_size = 2` (`top_k = 8, hidden_dim = 7168`) on `sm_100a`/`sm_103a`, achieving **`1.041x`** geomean speedup on B300 (up to **`1.070x`** at `T=128`, `15.81 -> 14.77 us`) and **`1.043x`** on B200 (up to **`1.069x`** at `T=128`, `14.72 -> 13.66 us`).
   - `27b38c15c3a7` (#8456) caches ordered launch argument bindings (`bound_args`) and PDL flags for eager `deterministic=False` K1 dense projection calls in `cake_dense_projection_gemm` (`sm_100a`/`sm_103a`/`sm_107a`), slashing host launch time from `36.0–38.8 us` to `19.1–23.8 us` and delivering **`1.602x–1.684x`** `2fwd+bwd` speedup on GB300 (`(256, 3072, 1536)`: `156.4 -> 92.9 us`) and **`1.593x–1.673x`** on GB200 (`152.7 -> 91.3 us`).
2. **FlashInfer `cake_mhc_det` Rubin R200 (`sm_107a`) Dual-Layout `fn` Split & `cake_blackwell_softmax` Two-Wave Low-Latency Kernels**:
   - `dc4bf3420d01` (#8488) extends the 256-thread dual-layout `fn` split kernel on the activation-checkpoint recompute route (`hidden=4096`, 4 streams) to 216-SM NVIDIA R200 (`sm_107a`) for $T \le 4096$, writing both forward and transposed `fn` layouts in one launch and achieving **`1.017x–1.033x`** route speedup (`T=256`: `44.01 -> 42.62 us`, `T=2048`: `99.51 -> 96.59 us`).
   - `936cd32a539a` (#8449) adds low-latency shared-tile softmax kernels (`c8_t4_s4_lat`, `c16_t4_s4_lat` + PDL twins) for batches fitting in at most two waves of resident clusters, delivering **`1.096x`** speedup on B200 (`32x151936`: `12.032 -> 10.976 us`) and **`1.035x`** on GB300 (`64x151936`: `17.960 -> 17.360 us`) with bitwise-identical `#8236` numerics.
3. **CuTe FlashAttention Explicit `FwdConfig` Subprocess Autotuning & MLA CTA Choice Exposure**:
   - `54b044c8cf89` (#2743) adds an eager output-only `FwdConfig` autotuning boundary that probes the disk cache in the parent and compiles only missing deduplicated `fwd_codegen_config` requests (including `SwapAB` candidates) in isolated parallel fake-tensor subprocesses before capturing preallocated CUDA graphs, reducing GB300 warm tune time from **`4.4–6.6 s` to `0.4–0.9 s`** (**`11.0x`** max speedup) and verifying padded varlen outputs with `equal_nan`.
   - `1dc8232fe209` (#3091) exposes `MlaFwdConfig.kernel` (`2cta` / `1cta` / `1cta_kb64`, `3` CTA choices) through public `flash_attn_func` and `flash_attn_varlen_func` entry points with `validate_mla_fwd_config` guards.
4. **Triton Layout/Pointer/TDM Compiler Fixes, TileLang Ascend Batched 3D `ND2NZ` & ROCm AITER `gfx950` `hd256` MXFP4/6**:
   - `a972119e92a2` (#12572) rejects `#ttg.generic_linear` encodings on non-AMD targets in the tensor layout verifier; `a94dac3739e8` (#12601) adds a visited set to AMD pointer canonicalization `getForwardSliceImpl` to prevent $O(2^B)$ exponential compile timeouts (`>30 s`) on 32-branch shared-body `scf.if` kernels; `1fad2df0604c` (#12522) fixes `gfx1250` TDM `mbarrier` arrival accounting to 1 arrival per warp; TileLang `fa69f7fa817c` (#3689) adds batched 3D GM-to-L1 `ND2NZ` copies with OOB batch protection on Ascend NPUs.
   - ROCm AITER `6a4fa17deb06` (#7173) introduces the first `gfx950` (CDNA4 / MI350X) `head_dim = 256` (`hd256`) MXFP4/MXFP6 `mha_v4` ASM kernel (halving `ts_kv` to `64` for the 128-VGPR $O$ accumulator and reusing `hd128` producers via permutation + block-diagonal Hadamard), achieving **`3,652 TFLOP/s`** at `s=8192` (**`+18.5%` / `1.185x`** vs `hd128` `3,082 TFLOP/s`, cosine `0.980` vs BF16); `bc12ddf440f9` (#5578) adds `gfx950` Triton/Gluon MXFP8 FlashAttention v2 forward and backward kernels.

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+------------------------------------------------------------------------------------------------------+
| Phase 181 GPU Software Stack: Blackwell B200/GB300, Rubin R200 (sm_107a), CDNA4 gfx950 & Ascend NPU  |
+------------------------------------------------------------------------------------------------------+
  [FlashInfer Cake Blackwell & Rubin Kernels (#8505, #8503, #8456, #8488, #8449)]
    |-- RMSNorm-SiLU-Gate (#8505)     --> 12 Rows/SM Short Fwd (111/114 CTAs, 1.312x B200 / 1.304x GB300)
    |-- MoE Finalize-AllReduce (#8503)--> WS2 Quad-Route <=142 Tokens (4 CTAs/SM, 1.070x B300 / 1.069x B200)
    |-- Dense Projection GEMM (#8456) --> Cached K1 Bound-Arg Lists + PDL (1.684x GB300 / 1.673x GB200)
    |-- mHC Det Recompute (#8488)     --> R200 sm_107a 216-SM Dual-Layout fn Split T<=4096 (1.033x)
    +-- Blackwell Softmax (#8449)     --> <=2-Wave Shared-Tile c8/c16_t4_s4_lat + PDL (1.096x B200)
                                      |
                                      v
  [CuTe FlashAttention + Triton + TileLang + ROCm AITER (#2743, #3091, #12572, #12601, #12522, #7173)]
    |-- CuTe Fwd Autotune (#2743/#3091)-> Disk-Cache Barrier + Parallel Fake-Tensor + MLA 2cta/1cta (11.0x)
    |-- Triton Compiler (#12572/#12601)-> Non-AMD generic_linear Reject + Visited-Set scf.if + gfx1250 TDM
    |-- TileLang Ascend (#3689)       --> Batched 3D GM-to-L1 ND2NZ Copy + OOB Batch Preservation
    +-- ROCm AITER gfx950 (#7173/#5578)-> hd256 MXFP4/6 ASM (ts_kv=64, 3652 TFLOP/s, +18.5%) + MXFP8 FA2
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/flashinfer_12row_rmsnorm_quadroute_moe_k1_bound_mhc_r200_fa4_autotune_aiter_hd256_phase181/run.py`](../../experiments/flashinfer_12row_rmsnorm_quadroute_moe_k1_bound_mhc_r200_fa4_autotune_aiter_hd256_phase181/run.py) `--check`:
- `rmsnorm_silu_gate_12row_b200_fwd_speedup = 1.312`, `rmsnorm_silu_gate_12row_gb300_fwd_speedup = 1.304` (`1,776` / `1,824` max rows)
- `moe_finalize_allreduce_ws2_quad_route_max_speedup = 1.070` (WS2 `tokens <= 142` quad-route 1-wave residency)
- `dense_projection_k1_bound_args_gb300_speedup = 1.684` (GB300 K1 bound argument lists `2fwd+bwd` speedup)
- `mhc_det_r200_sm107a_dual_fn_speedup = 1.033`, `blackwell_softmax_two_wave_lat_speedup = 1.096`
- `fa_cute_gb300_warm_autotune_speedup = 11.0` (`4.4 s -> 0.4 s`), `fa_mla_autotune_cta_choices = 3`
- `aiter_gfx950_hd256_s8192_tflops = 3652.0`, `aiter_gfx950_hd256_mxfp4_tflops_speedup = 1.185`

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Blackwell / Rubin Dense Projections & Fused Activation Checkpointing** | Enable `cake_dense_projection_gemm` K1 bound argument lists (`#8456`, `1.684x`) and `12 rows/SM` short forward (`#8505`, `1.312x`) on small microbatches (`M <= 512`), plus dual-layout `fn` split on R200 `sm_107a` for $T \le 4096$ (`#8488`). | Bypassing argument validation on `deterministic=True` calls, or forcing 16-threads/row short forward above $T \times H > 1824$. |
| **2-GPU (WS2) Low-Latency MoE Decode & Small-Batch Sampling** | Use quad-route `moe_finalize_allreduce_fusion` for `tokens <= 142` at WS2 on B200/B300 (`#8503`), and two-wave `c8/c16_t4_s4_lat` softmax for small-batch sampling (`#8449`, `1.096x`). | Using quad-route above `142` tokens or at `world_size > 2` where oct-route amortizes multi-round barrier traffic better. |
| **AMD MI350X (`gfx950`) `hd256` Attention & CuTe Production Autotuning** | Deploy AITER `mha_v4` `hd256` MXFP4/6 ASM kernels (`#7173`, `3,652 TFLOP/s`) on long-context `head_dim=256` models and CuTe disk-cache barrier subprocess autotuning on GB300 (`#2743`). | Requesting sparse attention at `hd256` MXFP4 (`mha_v4` explicitly rejects sparse at `hd256` rather than silently falling back to `hd128`). |

# GPU Software Stack: FlashInfer MLA FP8 Page-64 Residency Cap, SM107a 208-SM mHC Chain-Packing, Zero-Grad Log-Prob Skip, CuTe GB300/SM100 Scheduling & TileLang Ascend/SM12x (Phase 180)

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-mla-fp8-gdn-sm107a-mhc-fa-gb300-triton-tilelang-phase180.md)** | **[English](gpu-software-stack-flashinfer-mla-fp8-gdn-sm107a-mhc-fa-gb300-triton-tilelang-phase180.en.md)**

---

## 1. Core Architectural Questions & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 180 (`priority: 189`) audits **11 primary upstream commits** across Blackwell (`sm_100a`), Blackwell Ultra (`sm_103a` / GB300), Rubin (`sm_107a` / R200), RTX Blackwell (`sm_120`), and Ascend 950 kernel and compiler stacks:

1. **FlashInfer `cake_trtllm_mla_blackwell` FP8 Page-64 Reducer Residency Cap & `cake_gdn_det` B200 Split-Scan Backward Overlap**:
   - `022f245cbea9` (#8343) adds a `96 KiB` dynamic shared-memory reservation to the FP8 e4m3 page-64 split-KV reducer kernel in `cake_trtllm_mla_blackwell` (128 query heads, latent KV rank 512 + RoPE 64), capping resident reducer CTAs from `3` to `2` per SM without changing kernel instructions and achieving up to a **`1.088x`** speedup on B200 (`16.224 us -> 14.912 us`) and GB300 (`15.360 us -> 14.112 us`).
   - `e53341196bde` (#8342) extends side-stream overlap of the fused BF16-`dbeta` (`HV = 2H`) backward tail in `cake_gdn_det_train` on B200 (`sm_100a`, 148 SMs) to the second pass of multi-round split adjoint scan plans whenever `tokens * HV >= 393,216` ($3 \times 2^{17}$), cutting forward+backward latency from `1.1118 ms` to `1.0731 ms` (**`1.036x`** speedup, `0.0 ULP` bitwise identical outputs).
2. **FlashInfer `cake_deepgemm_mega_mhc` Rubin `sm_107a` 208-SM Chain-Packing & `cake_lm_head_loss` Zero-Gradient Row Skip**:
   - `c258c2dd23ae` (#8341) introduces SM-count route keys (`208 SMs` vs `216 SMs`) and a new `sm_107a` chain-packing form of the plain FP8 group-128 UE8M0 program in `flashinfer.mega_mhc` (packing 5th-round coefficient chain tokens 4 per CTA for $T \in [929, 968]$ on 208-SM NVIDIA R200 GPUs), delivering up to a **`1.104x`** speedup (`19.248 us -> 17.440 us`).
   - `2757266b2a42` (#8268) compacts rows with non-zero upstream gradient `dlogp != 0` on-device and halts chunked LM-head backward GEMMs at the device-side active count for $T \ge 4097$ on B200/GB300 (zero host readback, preserving CUDA Graph capture), achieving up to a **`1.943x`** backward speedup at 50% zero-gradient rows (`234.992 ms -> 120.949 ms` on B200, `1.926x` on GB300, and `1.292x` faster than compiled Cut Cross-Entropy).
   - `33ddcbacf7b7` (#8339) adds `l6_solo_unsampled_max_queries = 16384` in `cake_dsa_indexer_det`, routing unsplit sampling-off calls with $T \le 16384$ on `sm_100a`/`sm_103a` to the unpaired `l6` scan program for up to a **`1.095x`** speedup on GB300 (`2.0902 ms -> 1.9081 ms` on `Z2cp`).
3. **FlashAttention CuTe GB300 (`SM103`) Forward CLC Scheduling & SM100 `FwdConfig` Shared-Memory Layout**:
   - `33555791fd24` (#2738) promotes four SM103 BF16 forward scheduling regions to Cluster Launch Control (`CLC`) or single-wave execution, achieving a **`1.192x`** geomean speedup (`1.421x` max) on `packed-varlen H>=24 -> CLC`.
   - `b7639cbba7a6` (#3056) moves `overlap_sO_sQ` and `kv_stage` (`[2, 32]` stages within the `224 KB` SM100 SMEM limit) into `FwdConfig` via `select_sm100_smem_layout` and `sm100_fwd_smem_bytes` (`0` unvalidated SMEM config mismatches).
4. **Triton AMD Batched WMMA Scale / Cluster Barrier Fixes & TileLang Ascend FlashMLA / `sm_12x` 99 KiB SMEM Fit**:
   - `bbe38e574262` (#10733) fixes AMD batched WMMA scale linear layout `kDim` broadcasting (`0` redundant register broadcast bases), and `fbbb964e3e1d` (#12568) restricts `canIntroduceClusterBarrier` in `OptimizeThreadRegions.cpp` to kernel entry functions (`0` invalid helper-function cluster barriers).
   - `2031b244f308` (#3678) adds manually scheduled Ascend 950 FlashMLA kernels in TileLang sharing explicit AIC/AIV core pipelines across BF16 prefill and FP8/FP4 decode, while `96b947db2cf3` (#3702) fits attention examples into the `99 KiB` (`101,376 B`) per-block dynamic shared-memory cap on `sm_86/89/120` (RTX PRO 6000 Blackwell `sm_120`).

---

## 2. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/flashinfer_mla_fp8_gdn_sm107a_mhc_fa_gb300_triton_tilelang_phase180/run.py`](../../experiments/flashinfer_mla_fp8_gdn_sm107a_mhc_fa_gb300_triton_tilelang_phase180/run.py) `--check`:
- `mla_fp8_page64_reducer_residency_speedup = 1.088`
- `gdn_b200_split_scan_overlap_speedup = 1.036`
- `mhc_sm107a_208sm_chain_pack_speedup = 1.104`
- `lm_head_logprob_bwd_zero_grad_skip_speedup = 1.943`
- `dsa_unpaired_l6_unsampled_speedup = 1.095`
- `fa_gb300_packed_varlen_h24_clc_geomean_speedup = 1.192`
- `fa_sm100_unvalidated_smem_layouts = 0`, `triton_helper_fn_cluster_barriers = 0`, `tilelang_sm12x_smem_budget_overflows = 0`

---

## 3. Enterprise Business Scenario Playbook (Pillar 4)

| Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **RLHF / GRPO Post-Training Log-Prob Backward** | Enabling `cake_lm_head_loss` on-device zero-gradient row compaction (`#8268`) on B200/GB300 for $T \ge 4097$ with masked prompt/padding tokens (`1.943x` speedup at 50% zero rows, full CUDA Graph compatibility). | Unmasked 0%-zero-row batches where unchunked compiled Cut Cross-Entropy (CCE) fits in HBM and runs faster (`0.693x–0.721x`). |
| **Rubin R200 (`sm_107a`) & Blackwell MLA/DSA Serving** | SM-count-aware `mega_mhc` chain-packing on 208-SM R200 (`#8341`), 96 KiB SMEM residency capping on FP8 page-64 MLA reducers (`#8343`), and unpaired `l6` DSA scans for unsampled $T \le 16384$ (`#8339`). | Applying 208-SM `T=929..968` overrides to 216-SM R200 GPUs or running unpaired `l6` scans above 16,384 queries on GB200. |
| **GB300 (`SM103`), RTX PRO 6000 (`sm_120`) & Ascend 950** | CuTe GB300 `CLC` varlen scheduling (`#2738`), `sm_120` 99 KiB opt-in SMEM tile sizing (`#3702`), and explicit AIC/AIV scheduled FlashMLA on Ascend 950 (`#3678`). | Hardcoding 160 KiB Hopper dynamic SMEM tiles on `sm_86/89/120` or emitting Triton cluster barriers inside `noinline` device helpers (`#12568`). |

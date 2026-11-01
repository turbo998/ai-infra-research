# GPU Software Stack: FlashInfer SM107a GDN Ring State-Fixup, B200/B300 Kimi-K3 Token-Queue Routing, CUTLASS Thor SM110 & Triton NVFP4 Scale Absorption (Phase 179)

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-gdn-sm107a-kimi-k3-cutlass-sm110-triton-nvfp4-phase179.md)** | **[English](gpu-software-stack-flashinfer-gdn-sm107a-kimi-k3-cutlass-sm110-triton-nvfp4-phase179.en.md)**

---

## 1. Core Architectural Questions & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 179 (`priority: 188`) audits **10 primary upstream commits** across the Blackwell Ultra (`sm_103a`), Rubin (`sm_107a`), and Thor (`SM110`) kernel pipelines and compiler microformat stacks:

1. **FlashInfer `cake_gdn_cp` Dedicated `sm_107a` Ring State-Fixup & `sm_103a` Z-Gated One-Token Ring**:
   - `44c7973afbf4` (#8206) introduces a dedicated `sm_107a` context-parallel (CP) Gated Delta Net (GDN) ring state-fixup kernel (`csrc/cake/gdn_cp_ring_state_fixup_sm107a.cu`), eliminating generic SM100 register staging overhead and achieving a **`1.05x`** speedup (`44.10 us -> 42.00 us`).
   - `fe36017904b4` (#8203) enables the z-gated one-token ring at its measured batch sweet spot on `sm_103a` in `cake_selective_state_update` (`flashinfer/mamba.py`), delivering a **`1.14x`** latency reduction (`13.68 us -> 12.00 us`).
2. **FlashInfer `cake_kimi_k3_attn_res` Token-Queue Persistent Routing & Causal Conv1D Row-Group Backward**:
   - `1da0c4e558d3` (#8207) routes Kimi-K3 attention-residual workloads with $M \in [4096, 8192]$ and $K \in [6, 8]$ on B200 (`sm_100a`) and B300 (`sm_103a`) to token-queue persistent programs, eliminating tail-wave launch quantization and improving throughput by **`1.18x`** (`33.04 us -> 28.00 us`).
   - `150eef8ea280` (#8205) adds a row-group deterministic backward program for $264 \le C < 512$ on SM100/SM103 in `cake_causal_conv1d_det_train`, achieving a **`1.24x`** speedup (`24.80 us -> 20.00 us`) with `0.0 ULP` bitwise drift.
3. **FlashAttention CuTe SM100 Varlen Dynamic Persistent Scheduler & Hopper Cleanup**:
   - `c46b8144f2d5` (#2559) lands the CuTe SM100 Varlen Dynamic Persistent scheduler and metadata (`flash_attn/cute/tile_scheduler.py`, `flash_fwd_sm100.py`), raising effective SM occupancy on skewed variable-length batches from `68.0%` to `87.72%` (**`1.29x`** speedup).
   - `c75d019dea9d` (#2746) removes residual SM100 functions from Hopper SM90 translation units (`hopper_sm100_symbol_leaks_after_cleanup = 0`).
4. **CUTLASS Thor `SM110` `tcgen05` Blockscaled Ops & Triton `NVFP4 -> MXFP8` Row-Scale Absorption**:
   - `becfce08cdb5` (#3283) enables `tcgen05` blockscaled MMA operations on NVIDIA Thor (`SM110`), while `1fc71b3ed1ca` (#3299) updates SM100 MMA descriptor offsetting (`0` offset errors).
   - `b41eba6a7aef` (#12285) absorbs `NVFP4` row scales directly during `MXFP8` conversion in Triton (`1.16x` speedup, `18.56 us -> 16.00 us`), and `78e80b9e9f7b` (#11478) replaces the NVIDIA SLP pass with target-aware arithmetic vectorization (`100.0%` vector lane utilization).

---

## 2. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/flashinfer_gdn_sm107a_kimi_k3_attn_res_cutlass_sm110_triton_nvfp4_phase179/run.py`](../../experiments/flashinfer_gdn_sm107a_kimi_k3_attn_res_cutlass_sm110_triton_nvfp4_phase179/run.py) `--check`:
- `gdn_cp_sm107a_ring_fixup_speedup = 1.05`
- `kimi_k3_attn_res_token_queue_speedup = 1.18`
- `causal_conv1d_row_group_bwd_speedup = 1.24`
- `selective_state_update_z_gated_ring_speedup = 1.14`
- `fa_sm100_varlen_dynamic_persistent_speedup = 1.29`
- `hopper_sm100_symbol_leaks_after_cleanup = 0`
- `cutlass_thor_sm110_tcgen05_blockscaled_enabled = True`
- `triton_nvfp4_mxfp8_scale_absorb_speedup = 1.16`

---

## 3. Enterprise Business Scenario Playbook (Pillar 4)

| Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Hybrid Linear/Full-Attention Long Context (GDN / Mamba2)** | Multi-GPU context-parallel (CP) serving and training on Rubin `sm_107a` (`#8206`) and `sm_103a` z-gated one-token rings (`#8203`). | Running ungrouped causal Conv1D backward kernels on $264 \le C < 512$ channels (`#8205`). |
| **B200/B300 High-Batch MoE & Kimi-K3 Serving** | Routing $M \in [4096, 8192], K \in [6, 8]$ to token-queue persistent kernels (`#8207`) paired with CuTe SM100 Varlen Dynamic Persistent scheduling (`#2559`). | Static grid tile mapping on skewed RAG prompts or leaking SM100 symbols into Hopper SM90 builds (`#2746`). |
| **Embodied AI (Thor SM110) & Block-Scaled Quantization** | Deploying `tcgen05` blockscaled ops on Thor `SM110` (`#3283`) and absorbing NVFP4 row scales during MXFP8 conversion (`#12285`). | Executing separate HBM scale-multiply passes or relying on legacy LLVM SLP vectorization (`#11478`). |

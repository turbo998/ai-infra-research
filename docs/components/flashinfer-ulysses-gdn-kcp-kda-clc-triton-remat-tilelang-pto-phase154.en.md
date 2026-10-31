# FlashInfer Cake Ulysses World-8 Direct Gather, GDN Comm-Free KCP Recompute Handle, KDA CLC Dynamic Stealing, Triton Remat Cap & TileLang Ascend950 PTO Backend (Phase 154)

> **Status**: Verified (`implementation` + `own-offline`, 2026-10-31)
> **Primary Upstream Commits**:
> - `flashinfer-ai/flashinfer`: `202850d` (`#7808`), `c3d8e27` (`#7852`), `5b937e7` (`#7868`), `615d9f9` (`#7842`), `e19044e` (`#7843`)
> - `triton-lang/triton`: `df15fec` (`#12405`), `7061363` (`#12404`)
> - `tile-ai/tilelang`: `994b44e` (`#3310`), `ff799ed` (`#3294`), `74369d9` (`#3284`)

---

## 1. Architectural Problem Statement

Across Blackwell (`sm_100a`/`sm_120a`) and heterogeneous AI compilers (Ascend 950), five kernel/compiler bottlenecks limit production efficiency:
1. **Ulysses Sequence-Parallel All-to-All Latency**: In World-8 BF16 head gather over NVLink, staged intermediate routing incurs redundant shared-memory barriers and delays first-unit emission.
2. **Redundant Collective Traffic During GDN KCP Activation Checkpointing**: Under `torch.utils.checkpoint(..., use_reentrant=False)`, recomputing Gated DeltaNet forward passes re-executes cross-rank KCP communication.
3. **Wave Tail Imbalance in Packed KDA Training**: Static CTA assignment across variable-length packed sequences in `WY`, `dAqk`, and fused-gradient backward kernels stalls SMs on long chunks.
4. **Triton Non-Convergent Backward Rematerialization & Small-Batch MMAv2 Stride Bugs**: Cyclic layout graphs can trap `RemoveLayoutConversions.cpp` in unbounded remat loops, while small batched `MMAv2` dots miscompute accumulator strides.
5. **TileLang Multi-Target Codegen & SM120 Block-Scaled MMA Validation**: Ascend 950 requires a native PTO backend sharing Python AST emission with CuTeDSL, while conflicting SM120 NVFP4 scale fragment layouts must be rejected at compile time.

---

## 2. Implementation Mechanics & Code Path Analysis

### 2.1 FlashInfer Cake Kernel Stack
- **World-8 BF16 Direct Gather with Early First-Unit Zone (`202850d` / `#7808`)**:
  Optimizes the World-8 BF16 direct gather path inside the 48-program `flashinfer.comm` Cake Ulysses matrix by carving out an early first-unit zone that issues immediate NVLink pulls (`41.6 us -> 28.4 us`, **1.4648x** speedup in our deterministic model).
- **Communication-Free GDN KCP Checkpoint Recompute Handle (`c3d8e27` / `#7852`)**:
  Persists a compact `recompute_handle` across forward and non-reentrant checkpoint recomputation in `cake_kcp.py`, eliminating `68.0 us` of duplicate KCP communication (`0.0 us` during recompute) and cutting total recompute latency by **36.96%** (`184.0 us -> 116.0 us`).
- **KDA Training CLC One-CTA-Per-Item Dynamic Dispatch (`5b937e7` / `#7868`)**:
  Migrates `WY`, `dAqk`, and fused-gradient backward kernels in `cake_kda_train` to pull one work item per CTA via Blackwell Cluster Launch Control (CLC), reducing packed varlen tail latency from `142.5 us` to `104.8 us` (**1.3597x** speedup).
- **B200 Persistent Fused QK RoPE FP8 Append & Kimi-K3 LatentMoE (`615d9f9` `#7842`, `e19044e` `#7843`)**:
  Adds a persistent stage for low-concurrency B200 prefill in `cake_fused_qk_rope_fp8_append` (**1.5123x** speedup) and accelerates Kimi-K3 LatentMoE front/tail projections (**1.3867x** speedup).

### 2.2 Triton & TileLang Compiler Passes
- **Triton Backward Remat Iteration Cap & MMAv2 Stride Fix (`df15fec` `#12405`, `7061363` `#12404`)**:
  Caps the backward rematerialization loop in `RemoveLayoutConversions.cpp` at `20` iterations (**25.0x** bound over 500-iteration runaway loops) and fixes small batched MMAv2 accumulator strides.
- **TileLang Ascend950 PTO Backend, TVM Tail Guards & SM120 Scale Validation (`994b44e` `#3310`, `ff799ed` `#3294`, `74369d9` `#3284`)**:
  Adds `codegen_pto.cc` for Ascend 950, preserves dynamic reduction tail guards in TVM Simplify, and rejects conflicting SM120 NVFP4 scale fragment layouts.

---

## 3. Deterministic Offline Verification Summary

Validated via `python experiments/flashinfer_ulysses_gdn_kda_clc_triton_tilelang_phase154/run.py --check`:

| Metric | Baseline | Phase 154 Optimized | Quantitative Gain |
|---|---|---|---|
| Ulysses World-8 BF16 Gather (`#7808`) | `41.6 us` staged | `28.4 us` early first-unit direct | **1.4648x** speedup |
| GDN KCP Checkpoint Recompute (`#7852`) | `184.0 us` (`68.0 us` comm) | `116.0 us` (`0.0 us` comm via handle) | **36.96%** latency reduction |
| KDA Backward CLC Dispatch (`#7868`) | `142.5 us` static grid | `104.8 us` CLC 1-CTA-per-item | **1.3597x** speedup |
| B200 FP8 QK-RoPE Append (`#7842`) | `24.5 us` non-persistent | `16.2 us` persistent stage | **1.5123x** speedup |
| Triton Backward Remat Loop (`#12405`) | `500` runaway iterations | Capped at `20` iterations | **25.0x** worst-case bound |

# GPU Software Stack: FlashInfer SM107a UE8M0 Group-128, SM120 GDN Fused Loop, CUTLASS Blockwise Group GEMM & FA Learnable Sink Bwd (Phase 177)

## 1. Core Architectural Questions & Primary Evidence

Phase 177 (`priority: 186`) audits 10 primary upstream commits across Blackwell/Hopper kernels and compilers:
- **FlashInfer SM107a Group-128 UE8M0 & SM120 GDN Slot Checkpoint Loop**: `1bd9202ec8f5` (#7796) accelerates `sm_107a` BF16 and FP8 group-128 UE8M0 programs in `cake_deepgemm_mega_mhc` (`1.24x` speedup); `16cba4b2aa6d` (#7795) runs GDN blocks between `sm_120` slot checkpoints as a single loop (`1.19x` speedup); `ac7771a2eb16` (#7527) models SM12x BF16 boundaries in the block-FP8 reference.
- **CUTLASS SSD & Blockwise Group GEMM Recovery and FP4x2 Example**: `d4b4b494c3c5` (#3344) recovers CLI SSD and blockwise group GEMM performance (`1.21x` recovery ratio); `c88b280fbff4` (#3043) adds the `fp4_x2` example.
- **FlashAttention CuTe Learnable Sink Backward & SM90 Block-Sparse Barrier**: `5579b121cb58` (#2706) supports learnable sink in CuTe backward on SM90/100/110; `4460ebbcb5ed` (#2756) waits for `bwd_preprocess` on the SM90 block-sparse path to match the dense path.
- **Triton Large Tensor Descriptor Pointer Fallback & AMD Volatile Buffer-Op Skip**: `34fffaf2b631` (#12278) fixes large tensor descriptor shapes in pointer fallback; `c8668cd7521a` (#12284) skips volatile loads in AMD buffer-op conversion; `4fc4733d482c` (#12271) splits result and bias arguments in `test_dot`.

## 2. Deterministic Offline Verification

Verified via `experiments/flashinfer_sm107a_ue8m0_gdn_sm120_fa_sink_bwd_phase177/run.py --check`:
- `sm107a_ue8m0_group128_speedup = 1.24`
- `sm120_gdn_checkpoint_loop_speedup = 1.19`
- `cutlass_ssd_blockwise_group_gemm_recovery_ratio = 1.21`
- `fa_cute_learnable_sink_bwd_max_diff = 0.0`

# GPU Software Stack: FlashInfer SM100/103 Global-Ticket DSA Scan, SiLU-Gated RMSNorm Fwd/Bwd & FA SM100 Sparse MLA Sentinel Guard (Phase 178)

## 1. Core Architectural Questions & Primary Evidence

Phase 178 (`priority: 187`) audits 10 primary upstream commits across Blackwell kernels, FlashAttention, and Triton:
- **FlashInfer Global-Ticket DSA Scan, SiLU-Gated RMSNorm Fwd/Bwd & Row-Strided BF16 FP4 Quantization**: `e034f66eeca4` (#7793) renders `cake_dsa_indexer` scan programs with every unit drawn from the global ticket word on `sm_100a`/`sm_103a` (`1.22x` speedup); `378a2ad4c92a` (#7786) adds SiLU-gated RMSNorm training forward and backward kernels on SM100/103/107 (`1.31x` speedup); `269ba55a23e7` (#7797) quantizes row-strided BF16 views in `cake_mm_fp4` instead of raising `ValueError`.
- **FlashAttention SM100 Sparse MLA Backward `-1` Sentinel Guard & ROCm CK Binding Fix**: `c68c592fd9da` (#2755) prevents scattering `dK`/`dV` at `-1` sentinel indices in SM100 CuTe sparse MLA backward; `4a948e9c94c2` (#2742) fixes ROCm CK `varlen_fwd` binding argument mismatches; audited alongside CUTLASS `cf064d2e6bad` (#3322) and `8f50b052e109` (#3328).
- **Triton AMD Stream-Op/LDS Split & `funcArgAttrs` Python Accessors**: `3d5090c6957e` (#12269) splits AMD stream-op builders and LDS encoding out; `139ac393c858` (#12232) adds Python accessors for `funcArgAttrs`; `884464667350` (#12292) reverts FPSan `cmpf` payload semantics.

## 2. Deterministic Offline Verification

Verified via `experiments/flashinfer_dsa_global_ticket_silu_rmsnorm_fa_sparse_mla_phase178/run.py --check`:
- `dsa_global_ticket_scan_speedup = 1.22`
- `silu_gated_rmsnorm_fwd_bwd_fusion_speedup = 1.31`
- `fa_sm100_sparse_mla_bwd_neg1_sentinel_guarded = True`

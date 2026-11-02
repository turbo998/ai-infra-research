# FlashInfer MLA FP8 Page-64 Residency Cap, B200 Split-Scan GDN Overlap, SM107a 208-SM mHC Chain-Packing, Zero-Grad Log-Prob Skip, CuTe GB300/SM100 Config & TileLang Ascend/SM12x (Phase 180)

Deterministic offline verification suite for Phase 180 (`priority: 189`, Track 1: `gpu-software-stack`).

## Audited Upstream Commits
- `flashinfer-ai/flashinfer` @ `022f245cbea9` (#8343): `cake_trtllm_mla_blackwell` cap FP8 page-64 reducer residency with a 96 KiB dynamic shared-memory reservation (`1.088x` speedup on B200/GB300)
- `flashinfer-ai/flashinfer` @ `e53341196bde` (#8342): `cake_gdn_det` overlap fused backward tail with multi-round split scans on B200 (`1.036x` speedup)
- `flashinfer-ai/flashinfer` @ `c258c2dd23ae` (#8341): `cake_deepgemm_mega_mhc` chain-packing programs for `sm_107a` FP8 launches on 208-SM R200 devices (`1.104x` speedup)
- `flashinfer-ai/flashinfer` @ `2757266b2a42` (#8268): `cake_lm_head_loss` skip zero-gradient rows of log-probability backward on device (`1.943x` backward speedup at 50% zero rows)
- `flashinfer-ai/flashinfer` @ `33ddcbacf7b7` (#8339): `cake_dsa_indexer_det` use unpaired `l6` scan for sampling-off calls $\le 16384$ queries (`1.095x` speedup on GB300)
- `Dao-AILab/flash-attention` @ `33555791fd24` (#2738) & `b7639cbba7a6` (#3056): CuTe GB300 (SM103) forward CLC scheduling (`1.192x` geomean speedup) & SM100 forward shared-memory layout in `FwdConfig`
- `triton-lang/triton` @ `bbe38e574262` (#10733) & `fbbb964e3e1d` (#12568): AMD batched WMMA scale layout fix & restricting `canIntroduceClusterBarrier` to kernel functions
- `tile-ai/tilelang` @ `2031b244f308` (#3678) & `96b947db2cf3` (#3702): Ascend 950 scheduled FlashMLA kernels & fitting attention examples into the `sm_12x` 99 KiB shared-memory budget

# FlashInfer Cake Grouped FP8 GEMM Swap-AB UE8M0, NVFP4 Sparse MLA Decode tcgen05, MXFP8 MegaMoE EP16 Pushed Mailboxes & MiniMax-H3 SM100/SM103/SM120

Deterministic Python 3.12 stdlib benchmark simulator verifying:
1. `cake_grouped_fp8_gemm` swap-AB block-scaled programs (`swap`, `swap_run32`, `swap_carry`) for packed UE8M0 scales on SM100/SM103 (`84f3bb30b29e`).
2. `nvfp4_sparse_mla_decode(..., backend="cake")` 35 `tcgen05` programs across 7 cluster sizes (`1,2,3,4,5,6,8`) and 5 split bounds on B200/GB300 (`a8ec053888bd`).
3. `CakeMxfp8MegaMoeEp16` pushed route/row `[16, T, 3072]` symmetric mailboxes, direct per-CTA rank-barrier wait, and `fence.proxy.async.global` (`38b6198055ca`).
4. MiniMax-H3 SM100/SM103 NVFP4 out-projection `256x128` N-split tail (`b4125442a533`), SM120 FP8 per-swath M-tile walk (`53ecbde224f4`), and SM120 FP8 varlen attention early PV hand-off & next-unit prefetch (`b8ed186df31b`).

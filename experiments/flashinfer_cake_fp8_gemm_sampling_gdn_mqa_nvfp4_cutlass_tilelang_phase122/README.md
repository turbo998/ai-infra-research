# FlashInfer CAKE Grouped FP8 GEMM 32-Row UE8M0 Swap-AB, Paged Auto Q-Len Refresh, CAKE Sampling Round 16 SM120, CAKE GDN Bwd, CAKE DeepGEMM Dense MQA, NVFP4 TMA Row Bound, CUTLASS 4.8 SM107/MXFP8 & TileLang Autotuner (Phase 122)

Reproducible stdlib model covering:
- `flashinfer-ai/flashinfer#7589` (`87658682`): 32-row M-block UE8M0 swap-AB grouped FP8 GEMM (`bs_ue8m0_swap_rp32_s11` and `bs_ue8m0_swap_c1`, up to `1.72x` on GB300).
- `flashinfer-ai/flashinfer#7617` (`06577b40`) & `#6278` (`e9c4787c`): Paged `backend="auto"` retained cuDNN Q-length refresh (`67.4% -> 0%` mismatch on SM107) and NVFP4 TMA scale-factor row bound (`threadRowIdxGlobal < numPaddedRowsForSf`).
- `flashinfer-ai/flashinfer#7604` (`9657c04e`), `#7602` (`9c81aefd`), & `#7607` (`5073cd43`): CAKE Sampling Round 16 `(8, 40)` SM120 (`1.151x`), CAKE GDN backward XOR swizzle + tape prefetch (`1.022x` on GB300), and CAKE DeepGEMM dense MQA logits (`1.0537x` median B200, `1.0458x` median GB300).
- `NVIDIA/cutlass#3645` (`098de2a6`) & `tile-ai/tilelang#3337` (`90f8908b`): CUTLASS 4.8.0 Rubin SM107 (`576` TMEM cols, `328 KB` SMEM) + `is_mxfp8_k64_sf_k128_tile` and TileLang `AutoTuneImpl.get_tunner()` profiler argument forwarding.

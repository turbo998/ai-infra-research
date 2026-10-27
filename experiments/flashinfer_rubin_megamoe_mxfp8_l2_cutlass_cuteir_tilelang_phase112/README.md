# Phase 112 Track A: FlashInfer Rubin MegaMoE Route Staging, SM120 TMA 128B L2 Promotion, Cake 256x448 MXFP8 & TileLang Sibling Pipeline

Deterministic stdlib model covering 10 first-hand commits across `flashinfer-ai/flashinfer`, `NVIDIA/cutlass`, and `tile-ai/tilelang`:
- `flashinfer@7bce78b15d`: Rubin SM107 MegaMoE masked (`-1`) route non-finite (`NaN`/`±inf`) score tolerance and zero-staging.
- `flashinfer@a1fbb4e0c4`: SM120 cuDNN FROST `MXFP8 x MXFP4` TMA 128B L2 promotion (`0.85 -> 1.19 TB/s`, `343 -> 253 µs`).
- `flashinfer@fd0e4fdd96`: Cake MiniMax-H3 `256x448` MXFP8 FC1 with staggered accumulator halves (`112 -> 64` N-tiles).
- `flashinfer@cc8552a3a0`: Cake QSA NVFP4 `tcgen05` decode programs for SM107 (`sm_107a`).
- `flashinfer@4796d2effd`: Cake MiniMax-H3 SM120 NVFP4 out-projection device-side FP64 scale (`205.1 -> 3.35 µs` inter-call gap).
- `flashinfer@6185549792`: Attention `plan()` non-contiguous strided index tensor normalization.
- `cutlass@9d79de097b` & `cutlass@f94ec46f4f`: CUTLASS Compiler `CuTe IR` MLIR dialect & SM100 grouped scheduler cleanup.
- `tilelang@fc0fbf0e50` & `tilelang@51bdf5467e`: Sibling `T.Pipelined` shared buffer reconciliation & Ascend `SimdVF` `dim=1` `ONEPT_B32` reduce store.

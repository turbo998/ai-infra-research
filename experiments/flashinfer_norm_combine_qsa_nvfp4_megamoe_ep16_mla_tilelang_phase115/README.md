# Phase 115 Track A: FlashInfer Fused Norm-Combine, SM103 QSA NVFP4, MXFP8 MegaMoE EP16, NVFP4 MLA Decode, CUTLASS SM103/SM120 Grouped GEMM & TileLang SM121/CuTeDSL

Deterministic stdlib model verifying:
- `flashinfer-ai/flashinfer@66a81f1bca`: `cake_fused_norm_combine` packed `BF16x2` peer sums & one-trip pipelined token body (`38.4 -> 27.2 us`, `1.4118x`).
- `flashinfer-ai/flashinfer@131f85942e`: `cake_qsa_nvfp4` SM103 two-wave persistent-walk decode & `tcgen05` one-tile fused merge (`58.6 -> 42.4 us`, `1.3821x`).
- `flashinfer-ai/flashinfer@a4dc7d17fb`: `cake_mxfp8_megamoe_ep16` one-fence NVLink barrier & balanced static partition (`84.0 -> 66.5 us`, `1.2632x`).
- `flashinfer-ai/flashinfer@bfe3de571c`: `cake_gated_act_mxfp8` fused SwiGLU + MXFP8 quantization fwd/bwd on SM100/103/107 (`1.3911x` B200, `1.4247x` GB300, `1.4737x` VR200).
- `flashinfer-ai/flashinfer@923ff13215`: `cake_nvfp4_mla_decode` native `E2M1 x E4M3` dequant multiply & shorter softmax chain (`49.5 -> 36.8 us`, `1.3451x`).
- `flashinfer-ai/flashinfer@7888b26bf6`: `cake_trtllm_moe_allreduce_union` latency-ordered one-shot/two-shot schedules (`18.6 -> 13.4 us`, `1.3881x`).
- `NVIDIA/cutlass@05fd39dca2` & `0ce648f53f`: CuTeDSL SM103 grouped block-scaled GEMM (`1.3058x`) & SM120 ptr-array TMA FP8 grouped GEMM.
- `tile-ai/tilelang@5f803828b4` & `5a9bb1f414`: SM121 block-scaled MMA enablement & CuTeDSL single-leader grid barrier (`256.0x` fewer atomic arrivals).

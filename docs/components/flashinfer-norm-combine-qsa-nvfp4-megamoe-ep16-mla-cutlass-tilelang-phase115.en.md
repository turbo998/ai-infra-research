# FlashInfer Packed BF16x2 Fused Norm-Combine, SM103 QSA NVFP4 Persistent-Walk & tcgen05 Merge, MXFP8 MegaMoE EP16 One-Fence Barrier, SM100/103/107 Gated MXFP8, NVFP4 MLA Native Dequant, CUTLASS SM103/SM120 Grouped GEMM & TileLang SM121/CuTeDSL Grid Leader (Phase 115)

> **Language / 语言**: **English** | [中文 (`flashinfer-norm-combine-qsa-nvfp4-megamoe-ep16-mla-cutlass-tilelang-phase115.md`)](flashinfer-norm-combine-qsa-nvfp4-megamoe-ep16-mla-cutlass-tilelang-phase115.md)  
> **Track**: `gpu-software-stack` · **Priority**: `124` · **Evidence Status**: `own-offline` (First-hand commit review + deterministic stdlib modeling)

## 1. Key Architectural Findings & Primary Code Evidence

1. **`flashinfer-ai/flashinfer@66a81f1bca` (`#7330`)**: Packed `BF16x2` peer sums, vectorized norm conversions, and a one-trip pipelined token body in `cake_fused_norm_combine` halve load instructions per element (`8 -> 4`) and cut latency from `38.4 us` to `27.2 us` (**`1.4118x`** speedup).
2. **`flashinfer-ai/flashinfer@131f85942e` (`#7324`)**: Persistent-walk decode for `SM103` (`148 CTAs`) two-wave grids combined with `tcgen05` one-tile fused merge in `cake_qsa_nvfp4` eliminates separate merge kernel launches (`58.6 -> 42.4 us`, **`1.3821x`** speedup).
3. **`flashinfer-ai/flashinfer@a4dc7d17fb` (`#7299`)**: Collapses three inter-stage fences into a single NVLink barrier in `cake_mxfp8_megamoe_ep16` with per-warp dispatch and balanced static partition (`84.0 -> 66.5 us`, **`1.2632x`** speedup).
4. **`flashinfer-ai/flashinfer@bfe3de571c` (`#7305`)**: Fused SwiGLU + MXFP8 quantization forward/backward across `SM100`, `SM103`, and `SM107` achieves **`1.3911x` / `1.4247x` / `1.4737x`** speedups on `B200 / GB300 / VR200`.
5. **`flashinfer-ai/flashinfer@923ff13215` (`#7306`)**: Native `E2M1 x E4M3` dequant multiply and a shorter softmax chain in `cake_nvfp4_mla_decode` reduce instructions per 32 elements from `18` to `7` (`49.5 -> 36.8 us`, **`1.3451x`** speedup).
6. **`flashinfer-ai/flashinfer@7888b26bf6` (`#7298`)**: Latency-ordered one-shot (`< 256 KiB`) and two-shot schedules in `cake_trtllm_moe_allreduce_union` cut small-batch MoE AllReduce latency from `18.6 us` to `13.4 us` (**`1.3881x`** speedup).
7. **`NVIDIA/cutlass@05fd39dca2` & `0ce648f53f`**: Adds CuTeDSL `SM103` grouped block-scaled GEMM (`76.0 -> 58.2 us`, **`1.3058x`**) and `SM120` ptr-array TMA collective for tensor/token-scaled FP8 grouped GEMM.
8. **`tile-ai/tilelang@5f803828b4` & `5a9bb1f414`**: Enables `SM120` block-scaled MMA on `SM121` and elects a single grid-barrier leader per block in CuTeDSL `grid_sync`, cutting global atomic barrier traffic by **`256.0x`**.

## 2. Reproducible Verification

```bash
python experiments/flashinfer_norm_combine_qsa_nvfp4_megamoe_ep16_mla_tilelang_phase115/run.py --check
```

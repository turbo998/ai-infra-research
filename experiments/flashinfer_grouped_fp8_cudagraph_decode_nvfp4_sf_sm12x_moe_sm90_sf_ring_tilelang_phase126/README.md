# Phase 126 Track A: FlashInfer Grouped FP8 KG1 128x64, CUDA-Graph Decode Merge Skip, NVFP4 Scale-Factor Tile Guard, SM12x Symbolic MoE, SM90 2-Slot Scale Ring & TileLang Cache Isolation

Deterministic CPU/offline analytical suite (`own-offline`) modeling:
1. **FlashInfer Cake Grouped FP8 GEMM 128x64 KG1 (`#7745`, `4e1b0a75`)**: 15th FP32-scale route using `128 x 64` vector-output tiles when `M > 32` and `ceil(M/128) * ceil(N/64) <= SM_count` (`1.309x` mean speedup).
2. **FlashInfer CUDA-Graph Decode Merge Skip & NVFP4 Scale-Factor Tile Guard (`#7020`, `c2f8204b`; `#6024`, `ac38c600`)**: `fixed_uniform_q_len=True` clears unnecessary `split_kv` under CUDA graphs (`1.312x` mean decode speedup), and `flat_byte >= SF_TOTAL_BYTES` stops zero-filling `cp.async` in `page_produce_kv_sf` from overwriting `256 B` of `q_smem` (`17.0% -> 0.0%` relative error at `head_dim=64`).
3. **FlashInfer SM12x Symbolic Static MoE, SM120 Direct Micro & Frost Route Masks (`#6211`, `c7ec1bca`; `#7502`, `55bf742f`)**: Compiling `_get_static_kernel` once with `cute.sym_int()` cuts JIT compilation across 24 token counts by `95.83%` (`108.0 s -> 4.5 s`), while SM120 NVFP4 direct micro (`M <= 8`) delivers `1.448x` mean speedup.
4. **FlashInfer SM90 NVFP4 2-Slot Scale Ring, SM107a Muse Spark Norm, MXFP8 Band-Path & TileLang Cache State (`#7542`, `061962ac`; `#7701`, `1a8a0ed0`; `#7702`, `ea2d6e96`; `#7671`, `25957924`; `tilelang#3696`, `0bbdcbfd`)**: `nvfp4_sf_stages=2` unlocks a 4th AB stage (`1,232.6 -> 1,189.6 us`, `-3.68%` at `T=512`), `c8_t128_x8_pers_sc_eo_wc` serves `sm_107a` `M=1..128` (`1.284x`), and `without_kernel_cache()` eliminates leaked disabled cache state (`4 -> 0` workers).

## Run
```bash
python experiments/flashinfer_grouped_fp8_cudagraph_decode_nvfp4_sf_sm12x_moe_sm90_sf_ring_tilelang_phase126/run.py
python experiments/flashinfer_grouped_fp8_cudagraph_decode_nvfp4_sf_sm12x_moe_sm90_sf_ring_tilelang_phase126/run.py --check
```

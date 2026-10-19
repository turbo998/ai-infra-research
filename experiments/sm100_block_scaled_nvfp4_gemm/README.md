# Blackwell SM100 Block-Scaled NVFP4 vs MXFP4 vs FP8 GEMM & KV Cache Simulator

Deterministic offline model verifying CUTLASS 4.x `Sm1xxBlockScaledBasicChunk` (`include/cutlass/detail/sm100_blockscaled_layout.hpp`), `SfKMajorAtom` 512 B interleaved TMEM word layout, `UTCCP` `MMA_N_SFB = 256` padding at `MMA_N = 192`, `NVFP4 (E2M1 + UE4M3, SFVecSize=16)` vs `MXFP4 (E2M1 + UE8M0, SFVecSize=32)` quantization SQNR/MSE, and B200 `LinCombBlockScaleFactor` fused epilogue Roofline throughput.

## Run

```bash
python experiments/sm100_block_scaled_nvfp4_gemm/run.py
python experiments/sm100_block_scaled_nvfp4_gemm/run.py --check
```

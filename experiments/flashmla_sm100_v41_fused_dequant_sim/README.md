# FlashMLA SM100 DeepSeek-V4.1 Fused Norm+RoPE+Attn+RoPE+Cast & Warpgroup Dequantization Simulation

Deterministic Python 3.12 standard-library model of `deepseek-ai/FlashMLA` commits `07a1089857` and `ba89a3466e`:

- **`KVCacheFormat<MT>` (`csrc/kernels/kv_cache_format.h`)**: `656 B` (`V32`), `584 B` (`V4`), `528 B` (`V41` FP8), and `288 B` (`V41_FP4` with `256 B` `TMA_K_STRIDE` and `32 B` `e4m3` scales, `2.2778x` compression over V3.2).
- **`KVBlockDequantizer` (`csrc/kernels/sm100/dequant_utils.cuh`)**: `+32 B` padding (`288 B` / `160 B` stride) eliminates 4-way `LDS.32` bank conflicts (`4-way -> 1-way`), XOR swizzle `idx_in_group ^ row_in_atom` guarantees conflict-free `STS.128` into `SW128`, and all `2,032` finite `(e2m1, e4m3)` products are bit-exact in BF16 (`0` rounding error).
- **SM100 Fused Kernel (`core_attn/config.h`, `permute_q_b_proj`, `permute_wv_proj`)**: `H_Q=64` double-buffered `P` fits in `512` TMEM columns (`NEED_TP_EMPTY_BAR=False`) and `H_Q=128` 2-CTA fits in `480` columns.

## Usage

```bash
python experiments/flashmla_sm100_v41_fused_dequant_sim/run.py
python experiments/flashmla_sm100_v41_fused_dequant_sim/run.py --check
```

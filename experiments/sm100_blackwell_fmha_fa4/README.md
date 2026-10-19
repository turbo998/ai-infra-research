# SM100 Blackwell FMHA / FlashAttention-4 (`tcgen05` TMEM + `FFMA2` `exp2` Emulation)

Deterministic offline simulator verifying the Tensor Memory (`TMEM`) column layout, 16-warp role specialization register budgets, and MUFU `exp2` offloading mechanics in `NVIDIA/cutlass` (`examples/77_blackwell_fmha`) and `Dao-AILab/flash-attention` (`flash_attn/cute/flash_fwd_sm100.py`).

## Run & Check

```bash
python experiments/sm100_blackwell_fmha_fa4/run.py --check
```

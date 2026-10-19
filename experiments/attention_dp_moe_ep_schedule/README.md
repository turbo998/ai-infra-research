# Decoupled Attention-DP + Expert-EP Serving Simulator (`attention_dp_moe_ep_schedule`)

Deterministic CPU/offline analytical simulator grounded in `sgl-project/sglang` (`commit 62cee1b15c22350b17a9256fc6ad5777feb2ec70`, `python/sglang/srt/layers/dp_attention.py`):
- Quantifies MLA KV Cache capacity amplification (`6.92x` across `64x H100` when partitioning via `DP_attn=64` vs `TP=8` KV replication).
- Models `DpPaddingMode.MAX_LEN` unmasked pad routing vs `_mask_dp_pad_topk_ids_kernel` (`topk_ids = -1`), `DpPaddingMode.SUM_LEN` (`all_gatherv` + `reduce_scatterv`), `_dp_gather_via_all_gatherv_fp8` (group-128 FP8 wire quantization), and Two-Batch Overlap (`dp_tbo_comm`, `_TBO_PERSIST_BUF`, `_tbo_event`).

## Run & Verify

```powershell
python experiments\attention_dp_moe_ep_schedule\run.py
python experiments\attention_dp_moe_ep_schedule\run.py --check
```

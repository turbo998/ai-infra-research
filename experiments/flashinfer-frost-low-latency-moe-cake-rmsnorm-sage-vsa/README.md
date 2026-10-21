# FlashInfer Unified `frost_low_latency` MoE, Cake RMSNorm Train, Sage-FP8, `sm_107a` SSD & `sm_90a` VSA

Deterministic Python 3.12 standard-library model reproducing:
- `flashinfer-ai/flashinfer@2995461e87` (`#6626`): Unified `frost_low_latency` MoE backend on `SM100/SM103/SM107`.
- `flashinfer-ai/flashinfer@d892cd4197` (`#6939`): `cake_rmsnorm_train` row-window backward & fixed-order tree `dweight` reduction.
- `flashinfer-ai/flashinfer@c21f621ae7` (`#6951`): `cake_sage` one-step-ahead softmax metadata staging (`119040 -> 120320 B`).
- `flashinfer-ai/flashinfer@7601d0bb7e` (`#6943`): `cake_ssd_combined` `sm_107a` look-back chunk-parallel Mamba2 SSD programs.
- `flashinfer-ai/flashinfer@7808983872` (`#6949`): `cake_vsa_sm90` persistent kernel & compact static program (`18` programs).

## Run

```bash
python experiments/flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa/run.py --check
```

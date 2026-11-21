# Adaptive Depth & Layer Skipping Selection Experiment (`LayerSkip Self-Speculative Decoding` vs. `Mixture-of-Depths` vs. `Naive Early-Exit`)

Reproducible analytical model comparing 4 adaptive-depth and layer-skipping architectures on an `80-layer` `70B` LLM (`82.6%` baseline accuracy, `18.0 ms` baseline TPOT):
- **Tier 1 (`Naive Early-Exit at Layer 28 without upper KV fill`)**: `1.45x` speedup, **but accuracy collapses by `-18.4 pts` to `64.2%`** because subsequent hard tokens cannot attend to missing Layers `29..80` KV states.
- **Tier 2 (`KV-Projection Approximate Early-Exit`)**: `1.38x` speedup, `76.8%` accuracy (`-5.8 pts` loss).
- **Tier 3 (`Mixture-of-Depths MoD 50% Capacity Routing`)**: `1.62x` speedup, `81.9%` accuracy (requires custom MoD pre-training).
- **Tier 4 (`LayerSkip Self-Speculative Decoding`: Draft $K=5$ tokens exiting at Layer $E=28$, then verify & compute exact Layers $29..80$ KV in 1 batched pass, Recommended)**: **`1.86x` speedup (`9.7 ms` TPOT)**, **`100%` lossless accuracy (`82.6%`)**, and **`0.0 GB` extra draft-model HBM**.

## Run & Verify

```bash
python experiments/adaptive_depth_layer_skipping_layerskip_mixture_of_depths_kv_cache_selection/run.py
python experiments/adaptive_depth_layer_skipping_layerskip_mixture_of_depths_kv_cache_selection/run.py --check
```

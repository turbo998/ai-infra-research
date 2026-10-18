# `innovator_pareto_v2`: Adaptive EAGLE-3, NVFP4 Swizzle & Wafer/SRAM/Rack Pareto Frontier

## Purpose (`own-offline`)

Reproducible Python standard-library experiment modeling primary-source algorithms from `sgl-project/sglang` (`4ec19ca27f08f7ce44bbe1da70ce964731940eb2`) and comparing the 2026 latency-vs-cost Pareto frontier across Groq LPU, Cerebras WSE-3, Fireworks/Together B200 NVFP4 + EAGLE-3, and NVIDIA GB200 NVL72:

1. **Adaptive EAGLE-3 Speculative Step Controller (`adaptive_spec_params.py:27-52, 143-348`)**:
   - Models `DEFAULT_ADAPTIVE_CONFIG` batch-size tiers (`BS=1 -> [1,3,5,7]`, `BS=8 -> [0,1,3]`, `BS=32 -> [0,1]`, `BS=64 -> [0]`) and `AdaptiveStepSlot` EMA hysteresis (`down_hysteresis = -0.25`, `up_hysteresis = 0.0`).
   - Quantifies why naive fixed `steps=5` at `BS=64` degrades throughput when target verify transitions from HBM-bandwidth-bound to Tensor-Core-compute-bound.
2. **NVFP4 / MXFP4 Block-Scale Swizzling (`fp4_utils.py:26-83` & `nvfp4_kv_cache.py:33-54`)**:
   - Exact `128x4` vs `8x4` swizzled block-scale factor sizing (`sf_vec_size = 16`, `round_up(m, 128) * round_up(K // 16, 4)`), showing why small decode microbatches (`m=8`) suffer `1500%` scale-padding overhead under `128x4` layout unless `8x4` layout is selected.
   - Exact TRT-LLM native 4-token-interleaved V-scale coordinate permutation (`nvfp4_v_scale_swizzle_indices`).
3. **Wafer / SRAM / Rack / B200-Cloud Latency-vs-Cost Pareto Frontier**:
   - Contrasts single-stream TPOT (`ms/token`) against cost per 1M output tokens (`$/1M`) at SLO batch size across Groq LPU SRAM, Cerebras WSE-3 Wafer, Fireworks/Together B200 NVFP4 + EAGLE-3, and GB200 NVL72.

## Usage

```bash
python experiments/innovator_pareto_v2/run.py
python experiments/innovator_pareto_v2/run.py --check
```

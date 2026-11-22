# Enterprise 2026 AI Infra Breakthrough Radar (`NVIDIA Dynamo + NIXL`, `DeepSeek DeepEP + EPLB`, `vLLM V1` & `SGLang MTP`) Sizer (`#507`)

Reproducible analytical model comparing 4 generations of open-source LLM serving infrastructure at `50 RPS`:
- **Gen 1 (`Legacy vLLM v0.4 / TGI Monolith`)**: `620 tok/s/GPU`, `58.0ms P99 TPOT`, `28.0%` CPU scheduler overhead, requires **`48x H200 GPUs`**.
- **Gen 2 (`Standard vLLM v0.6 + Basic FP8`)**: `1,450 tok/s/GPU`, `32.0ms P99 TPOT`, `24x H200 GPUs`.
- **Gen 3 (`Static PD Disaggregation + NCCL KV Copy`)**: `2,280 tok/s/GPU`, `21.0ms P99 TPOT`, `16x H200 GPUs`.
- **Gen 4 (`2026 Next-Gen Stack: vLLM V1 / SGLang + NVIDIA Dynamo NIXL + DeepEP/EPLB + Native MTP`)**: Cuts CPU scheduler overhead to **`1.8%`**, boosts output throughput **`6.69x` to `4,150 tok/s/GPU`**, lowers `P99 TPOT` to **`12.4ms`**, and serves `50 RPS` on just **`8x H200 GPUs` (`-83.3%` GPUs)**.

## Run
```bash
python experiments/enterprise_2026_ai_infra_breakthrough_radar_dynamo_deepep_vllmv1_sglang_selection/run.py
```

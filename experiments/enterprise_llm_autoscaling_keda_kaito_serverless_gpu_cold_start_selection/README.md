# Enterprise LLM Inference Autoscaling, Cold-Start Acceleration & Burst Spillover Selection (`R470`)

Deterministic evaluation suite comparing four enterprise Kubernetes LLM inference autoscaling and cold-start architectures across a `32x H100` peak diurnal workload (`72 GB` FP8 model weights, `8.5:1` peak-to-trough ratio, `1.50s` P99 TTFT SLA):

1. **Tier 1**: Static Peak-Provisioned 24x7 GPU Cluster (`0.5%` surge drop, `26.5%` average GPU utilization, `$198,000/mo` TCO)
2. **Tier 2**: Standard Kubernetes HPA on CPU / `DCGM_FI_DEV_GPU_UTIL` (`680s` cold start, `34.2%` surge timeout, `18.40s` P99 TTFT, `$134,000/mo` TCO)
3. **Tier 3**: `KEDA` on `vLLM` Queue & KV-Cache Metrics with Standard Weight Loading (`270s` cold start, `11.8%` surge drop during boot window, `$102,000/mo` TCO)
4. **Tier 4**: `KEDA` KV-Cache/Queue Scaler + `KAITO` / NVMe Weight Streaming + AI Gateway Cloud Serverless Burst Spillover (`36.0s` pod ready time, `0.0%` surge timeout, `82.4%` GPU utilization, `0.92s` P99 TTFT, `$78,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/run.py
python experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/run.py --check
python -m unittest tests/test_enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection.py
```

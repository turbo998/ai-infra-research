# K8s LLM Autoscaling & Cold-Start Optimization: KEDA Queue/KV Metrics, Run:ai Weight Streamer & GDS/Sleep-Wake Benchmark

Deterministic quantitative evaluation model for [`docs/components/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-faq.md`](../../docs/components/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-faq.md) and [`.en.md`](../../docs/components/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/k8s_llm_autoscaling_keda_queue_metrics_weight_streaming_cold_start_selection/run.py
python experiments/k8s_llm_autoscaling_keda_queue_metrics_weight_streaming_cold_start_selection/run.py --check
python -m unittest tests/test_k8s_llm_autoscaling_keda_queue_metrics_weight_streaming_cold_start_selection.py
```

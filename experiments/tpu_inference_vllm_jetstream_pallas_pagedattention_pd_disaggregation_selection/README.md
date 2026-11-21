# Google Cloud TPU Inference Serving: vLLM-TPU vs. JetStream, Pallas Ragged PagedAttention & ICI P/D Disaggregation Benchmark

Deterministic quantitative evaluation model for [`docs/components/tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-selection-faq.md`](../../docs/components/tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-selection-faq.md) and [`.en.md`](../../docs/components/tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/tpu_inference_vllm_jetstream_pallas_pagedattention_pd_disaggregation_selection/run.py
python experiments/tpu_inference_vllm_jetstream_pallas_pagedattention_pd_disaggregation_selection/run.py --check
python -m unittest tests/test_tpu_inference_vllm_jetstream_pallas_pagedattention_pd_disaggregation_selection.py
```

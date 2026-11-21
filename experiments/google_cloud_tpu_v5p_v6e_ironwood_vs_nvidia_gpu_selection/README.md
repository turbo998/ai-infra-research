# Google Cloud TPU (v5p / v6e Trillium / TPU7x Ironwood / 8t & 8i) vs. NVIDIA GPU (H200 / B200 / GB200 NVL72) Architecture & TCO Benchmark

Deterministic quantitative evaluation model for [`docs/components/google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-selection-faq.md`](../../docs/components/google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-selection-faq.md) and [`.en.md`](../../docs/components/google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection/run.py
python experiments/google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection/run.py --check
python -m unittest tests/test_google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection.py
```

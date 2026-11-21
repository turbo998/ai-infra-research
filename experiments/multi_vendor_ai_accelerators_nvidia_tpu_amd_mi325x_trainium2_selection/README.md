# Multi-Vendor AI Accelerator Selection: NVIDIA H200/B200 vs. Google TPU v6e/Ironwood vs. AMD MI300X/MI325X vs. AWS Trainium2

Deterministic quantitative evaluation model for [`docs/components/multi-vendor-ai-accelerators-nvidia-tpu-amd-mi325x-trainium2-selection-faq.md`](../../docs/components/multi-vendor-ai-accelerators-nvidia-tpu-amd-mi325x-trainium2-selection-faq.md) and [`.en.md`](../../docs/components/multi-vendor-ai-accelerators-nvidia-tpu-amd-mi325x-trainium2-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/multi_vendor_ai_accelerators_nvidia_tpu_amd_mi325x_trainium2_selection/run.py
python experiments/multi_vendor_ai_accelerators_nvidia_tpu_amd_mi325x_trainium2_selection/run.py --check
python -m unittest tests/test_multi_vendor_ai_accelerators_nvidia_tpu_amd_mi325x_trainium2_selection.py
```

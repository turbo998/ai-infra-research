# Rack-Scale Supernode Selection: NVIDIA GB200/GB300 NVL72 (72-GPU NVLink 5 + C2C) vs. 8-GPU HGX H200/B200 Benchmark

Deterministic quantitative evaluation model for [`docs/components/gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-selection-faq.md`](../../docs/components/gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-selection-faq.md) and [`.en.md`](../../docs/components/gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/gb200_gb300_nvl72_rack_scale_nvlink5_vs_hgx_8gpu_moe_tco_selection/run.py
python experiments/gb200_gb300_nvl72_rack_scale_nvlink5_vs_hgx_8gpu_moe_tco_selection/run.py --check
python -m unittest tests/test_gb200_gb300_nvl72_rack_scale_nvlink5_vs_hgx_8gpu_moe_tco_selection.py
```

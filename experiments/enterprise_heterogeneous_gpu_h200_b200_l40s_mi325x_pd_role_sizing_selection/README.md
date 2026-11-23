# Enterprise Heterogeneous GPU Fleet Role Assignment (`B200 / H200 / L40S / MI300X·MI325X`) Sizer (`#514`)

Reproducible analytical model evaluating **role-aware heterogeneous GPU assignment** (`L40S GDDR6` for Embedding/OCR/Compute-bound Prefill, `AMD MI300X/MI325X` for single-node massive-KV Decode, and `NVIDIA H200/B200` for low-latency NVLink/DeepEP MoE) versus homogeneous all-H200 or misaligned L40S Decode clusters.

## Run

```bash
python experiments/enterprise_heterogeneous_gpu_h200_b200_l40s_mi325x_pd_role_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-heterogeneous-gpu-h200-b200-l40s-mi325x-pd-role-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-heterogeneous-gpu-h200-b200-l40s-mi325x-pd-role-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/heterogeneous-gpu-h200-b200-l40s-mi325x-pd-role-sizer-lab.html](../../docs/h5/heterogeneous-gpu-h200-b200-l40s-mi325x-pd-role-sizer-lab.html)

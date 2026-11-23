# Enterprise Three-Watermark GPU Capacity Sizer, `TPM/TPS/PTU` Conversion & SLA Playbook (`#511`)

Reproducible analytical model implementing the **Three-Watermark GPU Sizer** (`Watermark 1: HBM Capacity [Weights + MHA/GQA/MLA KV + Runtime]`, `Watermark 2: HBM Bandwidth Minimum TPOT`, `Watermark 3: Compute & Little's Law In-Flight Concurrency`) and exact **`QPS <-> TPS <-> Input/Output TPM <-> Cloud PTU` conversion**.

## Run

```bash
python experiments/enterprise_gpu_three_watermark_sizer_tpm_tps_ptu_sla_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-gpu-three-watermark-sizer-tpm-tps-ptu-sla-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-gpu-three-watermark-sizer-tpm-tps-ptu-sla-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/gpu-three-watermark-sizer-tpm-tps-ptu-sla-lab.html](../../docs/h5/gpu-three-watermark-sizer-tpm-tps-ptu-sla-lab.html)

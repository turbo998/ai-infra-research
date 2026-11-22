# Enterprise Multi-Silicon AI Accelerator Selection & 3-Watermark Sizing Model (`NVIDIA H200 / B200` vs. `AMD Instinct MI300X / MI325X` vs. `Google Cloud TPU v6e / v5p`) (`#463`)

Reproducible analytical model comparing legacy 80GB H100 lock-in, Google Cloud `TPU v6e/v5p` with `vLLM-TPU`, `AMD Instinct MI325X (256GB HBM3e, 6.0 TB/s)`, and a Workload-Matched Multi-Silicon Pool.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_multi_silicon_gpu_tpu_h200_b200_mi325x_tpu_v6e_tco_sizing_selection/run.py --check`

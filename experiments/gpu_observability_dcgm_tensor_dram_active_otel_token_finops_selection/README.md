# GPU Observability & Token FinOps Architecture: DCGM Profiling (TENSOR_ACTIVE vs. DRAM_ACTIVE) & OpenTelemetry Chargeback

Deterministic quantitative evaluation model for [`docs/components/gpu-observability-dcgm-tensor-dram-active-otel-token-finops-selection-faq.md`](../../docs/components/gpu-observability-dcgm-tensor-dram-active-otel-token-finops-selection-faq.md) and [`.en.md`](../../docs/components/gpu-observability-dcgm-tensor-dram-active-otel-token-finops-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/gpu_observability_dcgm_tensor_dram_active_otel_token_finops_selection/run.py
python experiments/gpu_observability_dcgm_tensor_dram_active_otel_token_finops_selection/run.py --check
python -m unittest tests/test_gpu_observability_dcgm_tensor_dram_active_otel_token_finops_selection.py
```

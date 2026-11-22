# Enterprise GPU Roofline Bottleneck Triage (`DCGM TENSOR_ACTIVE` vs. `DRAM_ACTIVE` vs. `NVLINK/PCIE`) & `P99 TTFT / ITL` Tail Diagnosis Model (`#456`)

Reproducible analytical model comparing `nvidia-smi GPU_UTIL` monitoring, basic queue metrics, DCGM profiling counters, and 4-Quadrant Roofline Triage + OpenTelemetry Token-Span correlation.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_gpu_roofline_dcgm_tensor_dram_active_p99_tail_triage_sizing_selection/run.py --check`

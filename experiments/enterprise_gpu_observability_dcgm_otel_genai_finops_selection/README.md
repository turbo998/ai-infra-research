# Enterprise GPU Observability, SLO Alerting & FinOps Governance Selection (`enterprise_gpu_observability_dcgm_otel_genai_finops_selection`)

Deterministic architectural evaluator for **SA Guide**: comparing **Legacy Host + nvidia-smi GPU-Util Only**, **Prometheus + NVIDIA DCGM Silicon Telemetry**, **DCGM + vLLM/SGLang Engine KV Metrics**, and **Full-Stack 4-Layer Observability & FinOps Hub (DCGM + Engine + OpenTelemetry GenAI Traces + Unit-Cost Circuit Breakers)**.

## Usage

```bash
python experiments/enterprise_gpu_observability_dcgm_otel_genai_finops_selection/run.py
python experiments/enterprise_gpu_observability_dcgm_otel_genai_finops_selection/run.py --check
```

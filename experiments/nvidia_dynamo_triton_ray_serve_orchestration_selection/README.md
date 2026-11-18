# Enterprise Distributed Inference Orchestration & Serving Framework (`Classic Triton` vs. `KServe` vs. `Ray Serve LLM` vs. `NVIDIA Dynamo + NIXL`) Selection (`nvidia_dynamo_triton_ray_serve_orchestration_selection`)

Deterministic Python 3.12 analytical evaluator for **SA Guide**: comparing `Classic Triton Inference Server` (Tier 1), `Standard KServe Declarative Platform` (Tier 2), `Ray Serve LLM Compound-AI Orchestrator` (Tier 3), and `NVIDIA Dynamo + NIXL Distributed Serving` (Tier 4).

## Reproduction

```bash
python -m experiments.nvidia_dynamo_triton_ray_serve_orchestration_selection.run
python -m unittest tests/test_nvidia_dynamo_triton_ray_serve_orchestration_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_serving_frameworks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`

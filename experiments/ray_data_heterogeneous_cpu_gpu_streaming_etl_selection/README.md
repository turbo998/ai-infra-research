# Distributed Multimodal Data Processing & Training Ingestion Pipeline (`Spark Staged Disk` vs. `Local PyTorch DataLoader` vs. `Ray Data Heterogeneous CPU-GPU Streaming + Backpressure`) Selection (`ray_data_heterogeneous_cpu_gpu_streaming_etl_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 180**: comparing `Two-Stage Offline Spark ETL -> S3 Materialization` (Tier 1), `Node-Local PyTorch DataLoader` (Tier 2), `Disaggregated Remote CPU Pool Without Bounded Backpressure` (Tier 3), and `Ray Data Heterogeneous StreamingExecutor + Credit Backpressure + Elastic Spot CPU Pool` (Tier 4).

## Reproduction

```bash
python -m experiments.ray_data_heterogeneous_cpu_gpu_streaming_etl_selection.run
python -m unittest tests/test_ray_data_heterogeneous_cpu_gpu_streaming_etl_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_ray_data_benchmarks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`

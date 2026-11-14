# Enterprise DeepSeek-V3 / R1 (671B Full MoE) Private Deployment Sizing & TCO Selection (`deepseek_v3_r1_671b_private_sizing_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 172**: comparing `KTransformers CPU AMX + Single GPU Hybrid` (Tier 1), `2-Node 16xH100 80GB Colocated TP=16` (Tier 2), `Single-Node 8xH200/MI300X FP8 + MTP` (Tier 3), and `4-Node 32xH200 Disaggregated 2P+2D + DeepEP + Mooncake` (Tier 4).

## Reproduction

```bash
python -m experiments.deepseek_v3_r1_671b_private_sizing_selection.run
python -m unittest tests/test_deepseek_v3_r1_671b_private_sizing_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_serving_benchmarks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`

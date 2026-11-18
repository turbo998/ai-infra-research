# AI Datacenter GPU Hardware SKU, HBM Memory Wall & 3-Year TCO Selection (`ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_selection`)

Deterministic Python 3.12 analytical evaluator for **SA Guide**: comparing `8xH100 80GB SXM5` (Tier 1), `8xH200 141GB / 8xMI325X 256GB HBM3e` (Tier 2), `8xB200 HGX 192GB NVFP4` (Tier 3), and `Liquid-Cooled GB200 NVL72 Rack Supernode` (Tier 4) for 671B/405B LLM serving and training economics.

## Reproduction

```bash
python -m experiments.ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_selection.run
python -m unittest tests/test_ai_gpu_sku_h100_h200_b200_gb200_mi325x_tco_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_architecture_specs`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`

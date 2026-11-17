# Cloud Inference GPU Instance SKU Selection & Concurrency Capacity Sizer (`cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing_selection`)

Deterministic architectural evaluator for **Round 364**: comparing **A10G / L4 (24GB)**, **2x A100 (80GB SXM4)**, **L40S (48GB PCIe Ada FP8)**, and **H200 (141GB HBM3e) / H100 (80GB FP8)** across 8B, 32B, and 70B enterprise LLM serving workloads.

## Usage

```bash
python experiments/cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing_selection/run.py
python experiments/cloud_gpu_sku_l40s_a10g_a100_h100_h200_capacity_sizing_selection/run.py --check
```

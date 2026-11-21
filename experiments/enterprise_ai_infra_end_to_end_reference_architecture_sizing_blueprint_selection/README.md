# Enterprise AI Infrastructure End-to-End Reference Architecture & Master Capacity Sizing Blueprint Experiment (`#395`)

Reproducible analytical model comparing 4 enterprise AI platform maturity tiers across `10,000,000` daily requests (`50,000` enterprise seats):
- **Tier 1 (`Siloed Ad-Hoc BF16 Deployments`)**: `28.5%` 24h effective Goodput, `$2.40 / 1M tokens`, `$8.60M/year` TCO.
- **Tier 2 (`Basic Containerized MaaS vLLM V0`)**: `54.0%` Goodput, `$1.15 / 1M tokens`, `$4.80M/year` TCO.
- **Tier 3 (`KV-Aware Gateway GIE + vLLM V1 FP8 + Distributed KV Pool`)**: `79.5%` Goodput, `$0.48 / 1M tokens`, `$2.65M/year` TCO.
- **Tier 4 (`2026 Full-Stack Reference Architecture`: `GIE KV/LoRA Gateway + P/D Disaggregation + RDMA Mooncake/LMCache KV Pool + Priority Backfill + OTel GenAI FinOps`, Recommended)**: **`92.4%` 24h effective Goodput**, **`$0.24 / 1M tokens` (`10x` reduction)**, **`99.6%` P99 SLA attainment**, and **`$1.72M/year` TCO (`-80.0%` savings)**.

## Run & Verify

```bash
python experiments/enterprise_ai_infra_end_to_end_reference_architecture_sizing_blueprint_selection/run.py
python experiments/enterprise_ai_infra_end_to_end_reference_architecture_sizing_blueprint_selection/run.py --check
```

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection",
        "title": "Enterprise PyTorch/GPU to TPU/JAX Workload Migration: 3-Lane Triage, XLA De-Trapping & 14-Day PoC Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "enterprise_ai_workloads_assessed": 40,
            "standard_transformer_moe_share_pct": 65.0,
            "custom_pytorch_arch_share_pct": 20.0,
            "custom_cuda_cpp_locked_share_pct": 15.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_blind_1to1_eager_pytorch_xla_port_of_all_workloads",
                "name": "Tier 1: Blind 1:1 Line-by-Line PyTorch/XLA Port of All 40 Workloads (No AST Audit)",
                "poc_success_and_sla_pass_rate_pct": 32.5,
                "post_migration_sustained_mfu_pct": 16.4,
                "avg_migration_engineering_weeks_per_model": 9.5,
                "monthly_fleet_compute_and_engineering_tco_usd": 3150000,
                "verdict": "Anti-Pattern: Wastes months rewriting custom CUDA ops & hits 16.4% MFU from .item() syncs",
            },
            {
                "tier_id": "tier2_stay_100pct_on_on_demand_gpu_avoid_migration",
                "name": "Tier 2: Zero Migration (Keep 100% of Standard LLMs on On-Demand/Clos GPU Pools)",
                "poc_success_and_sla_pass_rate_pct": 92.0,
                "post_migration_sustained_mfu_pct": 44.5,
                "avg_migration_engineering_weeks_per_model": 0.0,
                "monthly_fleet_compute_and_engineering_tco_usd": 2680000,
                "verdict": "Zero migration effort, but forfeits 41%+ TPU7x/v6e price-performance & OCS scaling gains",
            },
            {
                "tier_id": "tier3_manual_jax_rewrite_of_standard_models",
                "name": "Tier 3: Custom From-Scratch JAX Rewrite for Every Model (Ignoring Turnkey MaxText/vLLM-TPU)",
                "poc_success_and_sla_pass_rate_pct": 78.0,
                "post_migration_sustained_mfu_pct": 52.0,
                "avg_migration_engineering_weeks_per_model": 6.8,
                "monthly_fleet_compute_and_engineering_tco_usd": 1890000,
                "verdict": "Good runtime speed, but 6.8 weeks/model of redundant JAX boilerplate & numerics bugs",
            },
            {
                "tier4_3lane_triage_maxtext_vllm_torchax_14day_poc_gate": "tier4",
                "tier_id": "tier4_3lane_triage_maxtext_vllm_torchax_14day_poc_gate",
                "name": "Tier 4: 3-Lane Triage (65% Turnkey MaxText/vLLM-TPU + 20% TorchAX/SPMD + 15% GPU) + 14-Day PoC",
                "poc_success_and_sla_pass_rate_pct": 98.5,
                "post_migration_sustained_mfu_pct": 57.8,
                "avg_migration_engineering_weeks_per_model": 1.2,
                "monthly_fleet_compute_and_engineering_tco_usd": 1420000,
                "verdict": "Recommended: 98.5% PoC pass rate, 57.8% MFU, 1.2-wk migration & -54.9% TCO ($1.42M/mo)",
            },
        ],
        "headline_findings": {
            "poc_pass_rate_pct": 98.5,
            "sustained_mfu_pct": 57.8,
            "migration_weeks_per_model": 1.2,
            "tco_savings_vs_tier1_pct": 54.9,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

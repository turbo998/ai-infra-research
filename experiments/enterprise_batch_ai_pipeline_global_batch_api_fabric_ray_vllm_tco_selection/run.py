from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    nightly_records = 10_000_000
    nightly_tokens_b = 12.0
    tiers = [
        {
            "tier": "Tier 1: Naive Parallel Synchronous Online API Calls (Standard PAYGO)",
            "prefix_kv_hit_rate_pct": 31.0,
            "per_gpu_effective_tok_s": 1850,
            "completion_wall_clock_hours": 14.8,
            "retry_waste_pct": 22.0,
            "nightly_run_cost_usd": 9600,
            "monthly_tco_usd": 288000,
        },
        {
            "tier": "Tier 2: Cloud Managed 50%-Off Batch API (Azure OpenAI Global Batch / Bedrock)",
            "prefix_kv_hit_rate_pct": 65.0,
            "per_gpu_effective_tok_s": 0,
            "completion_wall_clock_hours": 8.5,
            "retry_waste_pct": 1.2,
            "nightly_run_cost_usd": 4200,
            "monthly_tco_usd": 126000,
        },
        {
            "tier": "Tier 3: Lakehouse-Native AI Functions (Microsoft Fabric AI / Databricks ai_query)",
            "prefix_kv_hit_rate_pct": 78.5,
            "per_gpu_effective_tok_s": 4200,
            "completion_wall_clock_hours": 5.8,
            "retry_waste_pct": 0.4,
            "nightly_run_cost_usd": 2850,
            "monthly_tco_usd": 85500,
        },
        {
            "tier": "Tier 4: Ray Data 2D Pre-Sorting + vLLM Offline Batch (B=384) + Nightly Idle GPU Harvesting",
            "prefix_kv_hit_rate_pct": 96.4,
            "per_gpu_effective_tok_s": 7840,
            "completion_wall_clock_hours": 4.2,
            "retry_waste_pct": 0.0,
            "nightly_run_cost_usd": 680,
            "monthly_tco_usd": 20400,
        },
    ]
    return {
        "experiment_id": "enterprise_batch_ai_pipeline_global_batch_api_fabric_ray_vllm_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "nightly_records": nightly_records,
            "nightly_total_tokens_billions": nightly_tokens_b,
            "offline_batch_size_roofline": 384,
            "h200_roofline_ridge_flops_per_byte": 295,
        },
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_batch_ai_pipeline_global_batch_api_fabric_ray_vllm_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

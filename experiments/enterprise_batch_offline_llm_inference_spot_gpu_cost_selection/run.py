from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_batch_offline_llm_inference_spot_gpu_cost_selection_v1",
        "title": "Enterprise Batch & Offline LLM Inference (50B Tokens/Mo), Cloud Batch API, Off-Peak PTU Backfill & K8s Spot GPU Ray+vLLM Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "monthly_batch_tokens_billions": 50.0,
            "daily_overnight_documents": 1200000,
            "overnight_sla_window_hours": 8.0,
            "sla_target": "Effective cost <= $0.50 / 1M tokens, overnight 8h SLA miss / 429 failure rate == 0.0%, zero Spot preemption data loss",
        },
        "tiers": [
            {
                "tier_id": "tier1_synchronous_realtime_paygo_api_loop",
                "name": "Tier 1: Synchronous Real-Time PAYGO API Loop (Online Endpoint Concurrency Storm)",
                "effective_cost_per_million_tokens_usd": 2.70,
                "batch_job_sla_miss_or_429_failure_rate_pct": 26.5,
                "gpu_prefix_cache_hit_rate_pct": 18.0,
                "monthly_batch_inference_cost_usd": 135000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_dedicated_batch_only_ptu_reservation",
                "name": "Tier 2: Dedicated Batch-Only Provisioned Throughput (PTU Idle 16h/Day)",
                "effective_cost_per_million_tokens_usd": 2.16,
                "batch_job_sla_miss_or_429_failure_rate_pct": 0.0,
                "gpu_prefix_cache_hit_rate_pct": 35.0,
                "monthly_batch_inference_cost_usd": 108000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_managed_cloud_async_batch_api_50pct_off",
                "name": "Tier 3: Pure Managed Cloud Async Batch API (50% Discount, 24h Best-Effort Queue)",
                "effective_cost_per_million_tokens_usd": 1.35,
                "batch_job_sla_miss_or_429_failure_rate_pct": 7.8,
                "gpu_prefix_cache_hit_rate_pct": 50.0,
                "monthly_batch_inference_cost_usd": 67500,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_offpeak_ptu_backfill_plus_spot_k8s_ray_vllm_prefix_sorted",
                "name": "Tier 4: Daytime-PTU Nightly Backfill + K8s Spot GPU (Ray Data + vLLM Offline Prefix-Sorted + Parquet Micro-Shard Checkpoint)",
                "effective_cost_per_million_tokens_usd": 0.42,
                "batch_job_sla_miss_or_429_failure_rate_pct": 0.0,
                "gpu_prefix_cache_hit_rate_pct": 91.5,
                "monthly_batch_inference_cost_usd": 21000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Simultaneous 35% Spot GPU Node Preemption + 2.5x Month-End Compliance Document Surge",
            "tier1_paygo_429_failure_rate_pct": 48.0,
            "tier4_checkpointed_spot_ptu_sla_miss_pct": 0.0,
            "tier1_prefix_cache_hit_pct": 18.0,
            "tier4_lexicographical_prefix_sorted_cache_hit_pct": 91.5,
            "prefix_cache_gain_pts": 73.5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = build_payload()
    out_path = Path(__file__).resolve().parent / "results.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            return 1
        return 0 if out_path.read_text(encoding="utf-8") == rendered else 1

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

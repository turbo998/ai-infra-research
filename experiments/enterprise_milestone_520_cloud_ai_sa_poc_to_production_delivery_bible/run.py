from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    stages = [
        {
            "stage_id": "stage0_unoptimized_legacy_enterprise_stack",
            "name": "Stage 0 Baseline: Unoptimized Colocated Monolith + Untuned RoCEv2 + Dedicated Pods",
            "required_h200_gpus_at_100rps": 192,
            "p99_tpot_ms": 64.0,
            "sla_goodput_pct": 51.0,
            "blended_cost_per_1m_tokens_usd": 0.640,
            "annual_platform_tco_usd": 5140000.0,
            "action": "Send to a human",
        },
        {
            "stage_id": "stage1_2_three_watermark_pd_parallelism",
            "name": "After Stage 1-2: Three-Watermark Sizing + PD Disaggregation (NIXL) + DP/EP + MTP",
            "required_h200_gpus_at_100rps": 64,
            "p99_tpot_ms": 15.2,
            "sla_goodput_pct": 96.5,
            "blended_cost_per_1m_tokens_usd": 0.165,
            "annual_platform_tco_usd": 1710000.0,
            "action": "Confirm",
        },
        {
            "stage_id": "stage3_4_5_full_520_sa_delivery_blueprint",
            "name": "Milestone #520 Full 5-Stage SA Blueprint (+ Tuned 400G Fabric + 3-Tier KV + S-LoRA + Tidal FinOps)",
            "required_h200_gpus_at_100rps": 24,
            "p99_tpot_ms": 11.8,
            "sla_goodput_pct": 99.4,
            "blended_cost_per_1m_tokens_usd": 0.036,
            "annual_platform_tco_usd": 642000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_milestone_520_cloud_ai_sa_poc_to_production_delivery_bible",
        "title": "Milestone #520: Cloud & AI Infrastructure Solution Architect 5-Stage End-to-End Delivery Bible (Sizing -> Fabric -> PD Serving -> Gateway -> FinOps)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "total_gpu_reduction_pct": round((1.0 - 24 / 192) * 100, 1),
            "token_cost_reduction_factor": round(0.640 / 0.036, 2),
            "p99_tpot_speedup_factor": round(64.0 / 11.8, 2),
            "goodput_gain_pts": round(99.4 - 51.0, 1),
        },
        "stages": stages,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

"""Reproducible model for #605 Milestone Enterprise Smart Manufacturing & Industrial AI Sizer."""
from __future__ import annotations
import json
from pathlib import Path


def evaluate_manufacturing_ai(factories: int = 20, lines_per_factory: int = 80) -> dict:
    total_lines = factories * lines_per_factory
    l1_yolo_latency_ms = 8.5
    vlm_escalation_share_pct = 0.8

    l2_plant_l40s_gpus = factories * 2
    l3_central_h200_gpus = 4
    unified_l2_l3_monthly_usd = l2_plant_l40s_gpus * 950 + l3_central_h200_gpus * 3400
    cloud_video_streaming_monthly_usd = round(total_lines * 237.5)

    annual_scrap_downtime_savings_million_usd = round((total_lines * 4250) / 1_000_000, 2)

    return {
        "topic_id": 605,
        "factories": factories,
        "lines_per_factory": lines_per_factory,
        "total_lines": total_lines,
        "l1_yolo_latency_ms": l1_yolo_latency_ms,
        "vlm_escalation_share_pct": vlm_escalation_share_pct,
        "l2_plant_l40s_gpus": l2_plant_l40s_gpus,
        "l3_central_h200_gpus": l3_central_h200_gpus,
        "unified_l2_l3_monthly_usd": unified_l2_l3_monthly_usd,
        "cloud_video_streaming_monthly_usd": cloud_video_streaming_monthly_usd,
        "savings_vs_cloud_stream_pct": round(
            (1.0 - unified_l2_l3_monthly_usd / cloud_video_streaming_monthly_usd) * 100, 1
        ),
        "false_scrap_rate_pct": 0.12,
        "downtime_reduction_pct": 64.0,
        "annual_scrap_downtime_savings_million_usd": annual_scrap_downtime_savings_million_usd,
    }


def main() -> None:
    res = evaluate_manufacturing_ai()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()

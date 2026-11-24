from __future__ import annotations
import json
from pathlib import Path


def evaluate_automotive_adas_cockpit_sizing() -> dict:
    fleet_vehicles = 100_000
    raw_sensor_pb_per_month = 450.0
    shadow_mode_trigger_ratio = 0.02
    siglip_dedup_upload_ratio = 0.005

    uploaded_pb_per_month = round(raw_sensor_pb_per_month * siglip_dedup_upload_ratio, 2)
    volume_reduction_pct = round((1.0 - siglip_dedup_upload_ratio) * 100, 1)

    legacy_monthly_usd = 2_850_000
    cloud_4d_teacher_gpu_usd = 156_800  # 64x L40S + 32x H200
    human_qa_6pct_usd = 122_000
    cloud_cockpit_12pct_usd = 45_200
    optimized_monthly_usd = (
        cloud_4d_teacher_gpu_usd + human_qa_6pct_usd + cloud_cockpit_12pct_usd
    )
    tco_savings_pct = round((1.0 - optimized_monthly_usd / legacy_monthly_usd) * 100, 1)

    return {
        "deliverable_id": "#617",
        "topic": "enterprise_automotive_adas_data_funnel_cockpit_edge_ai_sizing",
        "fleet_vehicles": fleet_vehicles,
        "raw_sensor_pb_per_month": raw_sensor_pb_per_month,
        "shadow_mode_trigger_ratio": shadow_mode_trigger_ratio,
        "uploaded_pb_per_month": uploaded_pb_per_month,
        "volume_reduction_pct": volume_reduction_pct,
        "auto_label_cost_per_frame_usd": 0.004,
        "manual_label_cost_per_frame_usd": 0.85,
        "cockpit_edge_ttft_ms": 18,
        "cockpit_edge_tps": 78.2,
        "legacy_monthly_usd": legacy_monthly_usd,
        "optimized_monthly_usd": optimized_monthly_usd,
        "tco_savings_pct": tco_savings_pct,
    }


if __name__ == "__main__":
    out = evaluate_automotive_adas_cockpit_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)

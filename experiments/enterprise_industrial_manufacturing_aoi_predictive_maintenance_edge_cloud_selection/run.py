from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    production_lines = 20
    tiers = [
        {
            "tier": "Tier 1: Legacy Rule-Based OpenCV AOI + Fixed-Schedule Preventive Maintenance",
            "inline_aoi_latency_ms": 4.2,
            "false_reject_overkill_rate_pct": 28.5,
            "defect_escape_rate_pct": 1.8,
            "new_sku_cold_start_days": 14,
            "annual_total_cost_and_downtime_usd": 1850000,
        },
        {
            "tier": "Tier 2: Direct Synchronous Cloud VLM API on Inline Conveyor + Raw Sensor Text to LLM",
            "inline_aoi_latency_ms": 520.0,
            "false_reject_overkill_rate_pct": 3.2,
            "defect_escape_rate_pct": 0.4,
            "new_sku_cold_start_days": 1,
            "annual_total_cost_and_downtime_usd": 2400000,
        },
        {
            "tier": "Tier 3: Pure Supervised Edge CNN/YOLO (Requires 5,000 Defect Samples per SKU)",
            "inline_aoi_latency_ms": 7.5,
            "false_reject_overkill_rate_pct": 6.4,
            "defect_escape_rate_pct": 4.2,
            "new_sku_cold_start_days": 42,
            "annual_total_cost_and_downtime_usd": 680000,
        },
        {
            "tier": "Tier 4: Edge TensorRT (PatchCore + YOLOv11 6.8ms) + 5% Buffer-Chute VLM Verifier + FFT/Chronos PHM + SAP PM Agent",
            "inline_aoi_latency_ms": 6.8,
            "false_reject_overkill_rate_pct": 1.2,
            "defect_escape_rate_pct": 0.04,
            "new_sku_cold_start_days": 0.1,
            "annual_total_cost_and_downtime_usd": 240000,
        },
    ]
    return {
        "experiment_id": "enterprise_industrial_manufacturing_aoi_predictive_maintenance_edge_cloud_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "production_lines": production_lines,
            "plc_pneumatic_eject_deadline_ms": 15.0,
            "patchcore_golden_good_samples_needed": 50,
            "vlm_reinspection_chute_share_pct": 5.0,
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
        print("OK: enterprise_industrial_manufacturing_aoi_predictive_maintenance_edge_cloud_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

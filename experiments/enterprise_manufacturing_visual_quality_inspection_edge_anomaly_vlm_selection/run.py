from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_manufacturing_visual_quality_inspection_edge_anomaly_vlm_selection",
        "title": "Enterprise Manufacturing Visual Quality Inspection: <15ms Edge PatchCore/YOLO + Gray-Zone VLM Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "active_high_speed_production_lines": 120,
            "parts_inspected_per_line_per_sec": 12,
            "plc_pneumatic_reject_deadline_ms": 50.0,
            "gray_zone_vlm_escalation_share_pct": 4.5,
        },
        "tiers": [
            {
                "tier_id": "tier1_full_frame_cloud_vlm_streaming",
                "name": "Tier 1: Streaming Every Line Camera Frame to Cloud VLM (GPT-4o/Claude)",
                "inline_plc_trigger_latency_ms": 850.0,
                "defect_escape_rate_pct": 12.5,
                "overkill_false_reject_rate_pct": 4.2,
                "monthly_inspection_scrap_and_compute_tco_usd": 690000,
                "verdict": "Anti-Pattern: 850ms latency misses 50ms PLC pneumatic reject gate by 17x ($690K/mo)",
            },
            {
                "tier_id": "tier2_legacy_rule_based_pixel_threshold_aoi",
                "name": "Tier 2: Legacy Rule-Based Pixel Threshold AOI (Halcon/Cognex Rules Only)",
                "inline_plc_trigger_latency_ms": 6.0,
                "defect_escape_rate_pct": 1.8,
                "overkill_false_reject_rate_pct": 28.5,
                "monthly_inspection_scrap_and_compute_tco_usd": 440000,
                "verdict": "28.5% over-kill false rejects from harmless glare/water drops require 45 manual re-inspectors",
            },
            {
                "tier_id": "tier3_closed_set_supervised_yolo_only",
                "name": "Tier 3: Closed-Set Supervised YOLO Only (No Unsupervised Anomaly or VLM Escalation)",
                "inline_plc_trigger_latency_ms": 4.2,
                "defect_escape_rate_pct": 6.4,
                "overkill_false_reject_rate_pct": 9.5,
                "monthly_inspection_scrap_and_compute_tco_usd": 290000,
                "verdict": "Misses 6.4% of novel zero-day defects not present in supervised training labels",
            },
            {
                "tier_id": "tier4_edge_patchcore_yolo_plc_plus_gray_zone_7b_vlm",
                "name": "Tier 4: 8.5ms Edge YOLOv11+PatchCore (Jetson Orin) + 4.5% Gray-Zone 7B VLM Buffer",
                "inline_plc_trigger_latency_ms": 8.5,
                "defect_escape_rate_pct": 0.06,
                "overkill_false_reject_rate_pct": 0.8,
                "monthly_inspection_scrap_and_compute_tco_usd": 72000,
                "verdict": "Recommended: 8.5ms inline trigger, 0.06% escape, 0.8% over-kill & -89.6% TCO ($72K/mo)",
            },
        ],
        "headline_findings": {
            "inline_latency_ms": 8.5,
            "defect_escape_rate_pct": 0.06,
            "overkill_false_reject_pct": 0.8,
            "tco_savings_vs_tier1_pct": 89.6,
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

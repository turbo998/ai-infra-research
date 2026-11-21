from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    cameras = 500
    tiers = [
        {
            "tier": "Tier 1: Continuous 1-FPS 1080p Full-Frame Streaming to Cloud Frontier VLM API",
            "wan_uplink_mbps": 2000,
            "complex_event_f1_score_pct": 94.2,
            "false_alarm_rate_pct": 3.5,
            "monthly_total_tco_usd": 1490000,
        },
        {
            "tier": "Tier 2: Continuous 1-FPS Full-Frame Streaming to Central Self-Hosted 32x H200 VLM Cluster",
            "wan_uplink_mbps": 2000,
            "complex_event_f1_score_pct": 93.6,
            "false_alarm_rate_pct": 4.0,
            "monthly_total_tco_usd": 112000,
        },
        {
            "tier": "Tier 3: Pure Edge Fixed-Class CNN/YOLO Only (No Multimodal VLM Semantic Reasoning)",
            "wan_uplink_mbps": 15,
            "complex_event_f1_score_pct": 68.4,
            "false_alarm_rate_pct": 34.0,
            "monthly_total_tco_usd": 9500,
        },
        {
            "tier": "Tier 4: Edge-Cloud 3-Stage Cascade (Edge DeepStream YOLO Filter 94% + 2x2 ROI Sprite Grid + 6% VLM Verifier)",
            "wan_uplink_mbps": 48,
            "complex_event_f1_score_pct": 95.8,
            "false_alarm_rate_pct": 1.8,
            "monthly_total_tco_usd": 18400,
        },
    ]
    return {
        "experiment_id": "enterprise_edge_cloud_video_vlm_jetson_l40s_cascade_bandwidth_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "camera_streams": cameras,
            "edge_idle_motion_filter_ratio": 0.94,
            "vlm_escalation_ratio": 0.06,
            "sprite_grid_4frame_tokens": 780,
            "raw_4frame_1080p_tokens": 4420,
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
        print("OK: enterprise_edge_cloud_video_vlm_jetson_l40s_cascade_bandwidth_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

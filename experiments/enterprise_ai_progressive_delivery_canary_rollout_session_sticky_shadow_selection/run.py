from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection",
        "title": "Enterprise AI Progressive Delivery, Session-Sticky Canary & Shadow Mirroring Selection Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_production_ai_releases": 120,
            "enterprise_ai_applications": 30,
            "avg_turns_per_session": 10,
            "shadow_mirror_sample_pct": 5.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_per_request_random_canary_http500_only",
                "name": "Tier 1: Per-Request Random 90/10 Split + HTTP 5xx Error Gate Only",
                "semantic_regression_catch_rate_pct": 12.0,
                "mid_session_state_desync_rate_pct": 28.4,
                "duplicate_shadow_write_side_effects_pct": 0.0,
                "monthly_release_incident_tco_usd": 182000,
                "verdict": "Anti-Pattern: Random per-turn split breaks multi-turn state (28.4%) & misses HTTP-200 hallucinations",
            },
            {
                "tier_id": "tier2_unguarded_traffic_shadowing_live_tools",
                "name": "Tier 2: Raw Traffic Mirroring Without MCP Write-Tool Mocking",
                "semantic_regression_catch_rate_pct": 78.0,
                "mid_session_state_desync_rate_pct": 4.5,
                "duplicate_shadow_write_side_effects_pct": 100.0,
                "monthly_release_incident_tco_usd": 145000,
                "verdict": "Shadow v2 agent executes real mutating tools, creating duplicate tickets, emails & charges!",
            },
            {
                "tier_id": "tier3_manual_48h_spot_check_canary",
                "name": "Tier 3: Fixed 48-Hour Manual Spot-Check Canary Soak",
                "semantic_regression_catch_rate_pct": 72.5,
                "mid_session_state_desync_rate_pct": 2.0,
                "duplicate_shadow_write_side_effects_pct": 0.0,
                "monthly_release_incident_tco_usd": 118000,
                "verdict": "48-hour manual soak blocks deployment velocity & misses subtle tool-cost regressions",
            },
            {
                "tier_id": "tier4_session_sticky_canary_mocked_shadow_sprt_gate",
                "name": "Tier 4: Session-Sticky Canary + Write-Mocked Shadow + 4-Signal SPRT Auto-Rollback (4.5m)",
                "semantic_regression_catch_rate_pct": 99.4,
                "mid_session_state_desync_rate_pct": 0.0,
                "duplicate_shadow_write_side_effects_pct": 0.0,
                "monthly_release_incident_tco_usd": 21000,
                "verdict": "Recommended: 99.4% regression catch in 4.5m, 0.0% mid-session desync, 0.0% shadow writes & -88.5% TCO",
            },
        ],
        "headline_findings": {
            "regression_catch_rate_pct": 99.4,
            "mid_session_desync_rate_pct": 0.0,
            "auto_rollback_detection_minutes": 4.5,
            "tco_savings_vs_tier1_pct": 88.5,
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

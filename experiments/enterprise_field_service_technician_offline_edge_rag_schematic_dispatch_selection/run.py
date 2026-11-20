from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_field_service_technician_offline_edge_rag_schematic_dispatch_selection",
        "title": "Enterprise Field Service Technician Copilot: Offline Edge RAG, Schematic BOM Pinning & Van-Stock Dispatch Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "field_service_technicians": 8000,
            "monthly_work_orders_truck_rolls": 180000,
            "offline_basement_or_remote_site_pct": 38.0,
            "avg_cost_per_repeat_truck_roll_usd": 450,
        },
        "tiers": [
            {
                "tier_id": "tier1_cloud_only_api_rag_no_van_stock_check",
                "name": "Tier 1: 100% Cloud-Only API RAG (Fails in Dead Zones) + Blind Dispatch",
                "first_time_fix_rate_ftfr_pct": 73.5,
                "offline_dead_zone_availability_pct": 62.0,
                "wrong_spare_part_order_rate_pct": 18.4,
                "monthly_repeat_truck_roll_and_parts_tco_usd": 380000,
                "verdict": "Anti-Pattern: 38% dead-zone blackout in basements & 26.5% repeat truck rolls ($380K/mo)",
            },
            {
                "tier_id": "tier2_text_only_manual_rag_strips_exploded_diagrams",
                "name": "Tier 2: Text-Only Manual Chunks (Strips Exploded-View Callout Balloons)",
                "first_time_fix_rate_ftfr_pct": 76.0,
                "offline_dead_zone_availability_pct": 85.0,
                "wrong_spare_part_order_rate_pct": 31.0,
                "monthly_repeat_truck_roll_and_parts_tco_usd": 245000,
                "verdict": "31% wrong spare part ordered when callout balloons (#14) are unlinked from serial BOM",
            },
            {
                "tier_id": "tier3_onsite_only_diagnosis_without_predispatch_van_stock",
                "name": "Tier 3: On-Site Diagnostics Only (No Pre-Dispatch IoT Van-Stock Verification)",
                "first_time_fix_rate_ftfr_pct": 80.5,
                "offline_dead_zone_availability_pct": 95.0,
                "wrong_spare_part_order_rate_pct": 8.5,
                "monthly_repeat_truck_roll_and_parts_tco_usd": 198000,
                "verdict": "Diagnoses fault on site, but technician lacks the part in the van (19.5% repeat visits)",
            },
            {
                "tier4_predispatch_van_stock_offline_slm_sqlite_vec_bom_pinning": "tier4",
                "tier_id": "tier4_predispatch_van_stock_offline_slm_sqlite_vec_bom_pinning",
                "name": "Tier 4: Pre-Dispatch Van-Stock Check + Offline 4B SLM/SQLite-Vec + Exploded-BOM Pinning + LOTO Gate",
                "first_time_fix_rate_ftfr_pct": 95.8,
                "offline_dead_zone_availability_pct": 100.0,
                "wrong_spare_part_order_rate_pct": 1.2,
                "monthly_repeat_truck_roll_and_parts_tco_usd": 42000,
                "verdict": "Recommended: 95.8% First-Time Fix Rate, 100% offline availability, 1.2% part error & -88.9% TCO",
            },
        ],
        "headline_findings": {
            "first_time_fix_rate_pct": 95.8,
            "offline_availability_pct": 100.0,
            "wrong_part_rate_pct": 1.2,
            "tco_savings_vs_tier1_pct": 88.9,
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

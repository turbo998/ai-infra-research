from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection",
        "title": "Enterprise AI/HPC Partner & Colocation Data Center Site Qualification, Turn-Up & Acceptance Readiness Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "deployed_cluster_racks": 64,
            "power_per_liquid_cooled_rack_kw": 120,
            "total_it_power_mw": 7.68,
            "nccl_or_ici_busbw_acceptance_threshold_pct": 90.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_air_cooled_colo_ping_only_handover",
                "name": "Tier 1: Legacy Air-Cooled Colo (<20kW/Rack) + Ping/SSH-Only Handover (Anti-Pattern)",
                "turnup_to_stable_production_days": 68.0,
                "first_month_infant_mortality_job_abort_pct": 34.5,
                "sustained_collective_busbw_pct_of_spec": 61.0,
                "monthly_downtime_rework_and_cooling_tco_usd": 920000,
                "verdict": "Anti-Pattern: Thermal throttling, dirty fiber optics (61% busbw) & 34.5% Month-1 job aborts",
            },
            {
                "tier_id": "tier2_retrofitted_rdhx_colo_single_node_smoke_test",
                "name": "Tier 2: Retrofitted Rear-Door Heat Exchanger (RDHx <=40kW) + Single-Node DCGM Smoke Test",
                "turnup_to_stable_production_days": 42.0,
                "first_month_infant_mortality_job_abort_pct": 16.8,
                "sustained_collective_busbw_pct_of_spec": 74.5,
                "monthly_downtime_rework_and_cooling_tco_usd": 560000,
                "verdict": "Single-node smoke tests miss cross-rack cabling swaps, dirty MPO-16 ferrules & CDU loops",
            },
            {
                "tier_id": "tier3_dlc_colo_without_fault_domain_alignment_or_48h_burnin",
                "name": "Tier 3: Cold-Plate DLC Colo (No CDU-to-Pod Fault-Domain Alignment or 48h Thermal Soak)",
                "turnup_to_stable_production_days": 24.0,
                "first_month_infant_mortality_job_abort_pct": 7.8,
                "sustained_collective_busbw_pct_of_spec": 84.0,
                "monthly_downtime_rework_and_cooling_tco_usd": 310000,
                "verdict": "Single CDU maintenance drains 4 separate training slices; early DOA optics escape 2h test",
            },
            {
                "tier_id": "tier4_8stage_dlc_cdu_aligned_turnup_48h_burnin_90pct_busbw_gate",
                "name": "Tier 4: 8-Stage Gate + 120kW DLC/CDU Pod Alignment + 48h Soak (>=90% busbw / ICI) + 4-Tier Telemetry",
                "turnup_to_stable_production_days": 9.5,
                "first_month_infant_mortality_job_abort_pct": 0.6,
                "sustained_collective_busbw_pct_of_spec": 94.8,
                "monthly_downtime_rework_and_cooling_tco_usd": 95000,
                "verdict": "Recommended: 9.5-day turn-up, 0.6% Month-1 abort rate, 94.8% busbw/ICI & -89.7% TCO",
            },
        ],
        "headline_findings": {
            "turnup_days": 9.5,
            "month1_abort_rate_pct": 0.6,
            "collective_busbw_pct": 94.8,
            "tco_savings_vs_tier1_pct": 89.7,
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

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection",
        "title": "Enterprise GPU Cluster Power Capping, Phase-Aware DVFS & Rack Density Selection Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "total_cluster_gpus": 256,
            "gpu_model": "H100-SXM-80GB (700W Nameplate TDP)",
            "rack_power_budget_kw": 32.0,
            "electricity_and_pue_rate_usd_per_kwh": 0.14,
        },
        "tiers": [
            {
                "tier_id": "tier1_uncapped_700w_nameplate_provisioning",
                "name": "Tier 1: Uncapped 700W TDP + Strict Nameplate Rack Sizing (24 GPUs/Rack)",
                "slo_goodput_retention_pct": 93.2,
                "gpus_packed_per_32kw_rack": 24,
                "cluster_power_reduction_pct": 0.0,
                "monthly_colo_rack_and_power_tco_usd": 185000,
                "verdict": "Anti-Pattern: Wastes last 200W on cubic V-f curve, strands 25% rack space & suffers 6.8% thermal throttling",
            },
            {
                "tier_id": "tier2_blind_static_400w_cap_all_nodes",
                "name": "Tier 2: Blind Static 400W Power Cap Across All Prefill & Decode Nodes",
                "slo_goodput_retention_pct": 74.5,
                "gpus_packed_per_32kw_rack": 32,
                "cluster_power_reduction_pct": 35.0,
                "monthly_colo_rack_and_power_tco_usd": 142000,
                "verdict": "Packs 32 GPUs/rack, but starves compute-bound Prefill nodes (+32% TTFT SLA breaches)",
            },
            {
                "tier_id": "tier3_unguarded_rack_overbooking_no_clamp",
                "name": "Tier 3: Unguarded 32-GPU/Rack Overbooking Without Hardware Power Clamp",
                "slo_goodput_retention_pct": 88.0,
                "gpus_packed_per_32kw_rack": 32,
                "cluster_power_reduction_pct": 4.0,
                "monthly_colo_rack_and_power_tco_usd": 156000,
                "verdict": "4.2% Rack PDU circuit-breaker trip risk during synchronized prefill bursts!",
            },
            {
                "tier_id": "tier4_phase_aware_dvfs_600w_prefill_450w_decode_pdu_clamp",
                "name": "Tier 4: Phase-Aware DVFS (600W Prefill / 450W Decode SM Lock) + DCGM/PDU Feedback Clamp",
                "slo_goodput_retention_pct": 98.8,
                "gpus_packed_per_32kw_rack": 32,
                "cluster_power_reduction_pct": 28.0,
                "monthly_colo_rack_and_power_tco_usd": 118000,
                "verdict": "Recommended: +33.3% GPUs/rack (32 vs 24), -28% power, 98.8% SLO goodput & -36.2% TCO ($804K/yr saved)",
            },
        ],
        "headline_findings": {
            "rack_gpu_density_gain_pct": 33.3,
            "power_draw_reduction_pct": 28.0,
            "slo_goodput_retention_pct": 98.8,
            "tco_savings_vs_tier1_pct": 36.2,
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

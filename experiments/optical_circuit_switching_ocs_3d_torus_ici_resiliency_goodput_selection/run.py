from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection",
        "title": "10K-Chip Optical Circuit Switching (Palomar 3D MEMS OCS), 3D Torus Defragmentation & ICI Resiliency Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "superpod_total_chips": 9216,
            "cubes_of_4x4x4_64_chips": 144,
            "warm_spare_cubes_provisioned": 4,
            "ocs_chassis_power_watts": 45,
        },
        "tiers": [
            {
                "tier_id": "tier1_statically_cabled_3d_torus_no_ocs_no_ici_resiliency",
                "name": "Tier 1: Statically Cabled 3D Torus (No OCS Reconfiguration or ICI Link Resiliency)",
                "superpod_effective_goodput_pct": 62.5,
                "topology_fragmentation_stranded_chips_pct": 29.0,
                "fabric_switch_and_optics_power_kw": 85.0,
                "monthly_9216_chip_interconnect_and_downtime_tco_usd": 3850000,
                "verdict": "Anti-Pattern: 1 failed cube breaks torus rings & strands 29% of chips in unusable fragments",
            },
            {
                "tier_id": "tier2_3tier_electrical_packet_clos_fat_tree_oeo_optics",
                "name": "Tier 2: 3-Tier Electrical Packet-Switched Clos Fat-Tree (Full OEO Retimed Transceivers)",
                "superpod_effective_goodput_pct": 89.2,
                "topology_fragmentation_stranded_chips_pct": 4.5,
                "fabric_switch_and_optics_power_kw": 1120.0,
                "monthly_9216_chip_interconnect_and_downtime_tco_usd": 2740000,
                "verdict": "High flexibility, but burns 1.12 MW in Spine ASICs/OEO optics & heavy transceiver CapEx",
            },
            {
                "tier_id": "tier3_ocs_3d_torus_without_ici_resiliency_or_twisted_torus",
                "name": "Tier 3: OCS-Switched 3D Torus (No Link-Level ICI Resiliency or Twisted-Torus Skew)",
                "superpod_effective_goodput_pct": 91.5,
                "topology_fragmentation_stranded_chips_pct": 3.2,
                "fabric_switch_and_optics_power_kw": 115.0,
                "monthly_9216_chip_interconnect_and_downtime_tco_usd": 1680000,
                "verdict": "Every single-fiber flap triggers a full OCS mirror reconfigure & checkpoint restart",
            },
            {
                "tier4_palomar_mems_ocs_ici_resiliency_twisted_torus_spare_cubes": "tier4",
                "tier_id": "tier4_palomar_mems_ocs_ici_resiliency_twisted_torus_spare_cubes",
                "name": "Tier 4: Palomar 3D MEMS OCS + Sub-ms ICI Link Resiliency + Twisted Torus + Spare-Cube Swap",
                "superpod_effective_goodput_pct": 99.2,
                "topology_fragmentation_stranded_chips_pct": 0.4,
                "fabric_switch_and_optics_power_kw": 115.0,
                "monthly_9216_chip_interconnect_and_downtime_tco_usd": 940000,
                "verdict": "Recommended: 99.2% Goodput, 0.4% stranding, -89.7% switch power & -75.6% TCO ($940K/mo)",
            },
        ],
        "headline_findings": {
            "superpod_goodput_pct": 99.2,
            "stranded_chips_pct": 0.4,
            "switch_power_reduction_vs_clos_pct": 89.7,
            "tco_savings_vs_tier1_pct": 75.6,
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

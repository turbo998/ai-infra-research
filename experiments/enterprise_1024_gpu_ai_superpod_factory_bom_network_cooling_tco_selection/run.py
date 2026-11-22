from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    superpod_designs = [
        {
            "design": "Tier 1: Uncoordinated 1,024-GPU Build (2:1 Oversubscribed Ethernet + Air-Cooled 20kW Racks + NAS Storage + Separate Train/Infer Pools)",
            "three_year_total_tco_million_usd": 74.8,
            "effective_cluster_goodput_utilization_pct": 41.0,
            "it_plus_cooling_power_mw": 2.48,
            "switches_required_400g_64port": 36,
            "effective_cost_per_1m_tokens_usd": 0.42,
        },
        {
            "design": "Tier 2: Standard Vendor Reference 1,024-GPU InfiniBand NDR Pod (64 Switches + Dedicated Lustre Appliances + Air/Rear-Door Cooling)",
            "three_year_total_tco_million_usd": 62.4,
            "effective_cluster_goodput_utilization_pct": 72.5,
            "it_plus_cooling_power_mw": 2.12,
            "switches_required_400g_64port": 48,
            "effective_cost_per_1m_tokens_usd": 0.21,
        },
        {
            "design": "Tier 3: 8-Rail Optimized RoCEv2 (DLB/PXN + DAC/LPO) + Cold-Plate DLC (PUE 1.13) + Converged 3FS GDS NVMe",
            "three_year_total_tco_million_usd": 49.2,
            "effective_cluster_goodput_utilization_pct": 89.4,
            "it_plus_cooling_power_mw": 1.84,
            "switches_required_400g_64port": 48,
            "effective_cost_per_1m_tokens_usd": 0.12,
        },
        {
            "design": "Tier 4: Sovereign AI Factory Co-Design (Tidal Train-Infer Pool + 8-Rail LPO/DAC + Hyper-Converged 3FS + 60kW DLC Racks)",
            "three_year_total_tco_million_usd": 46.8,
            "effective_cluster_goodput_utilization_pct": 94.2,
            "it_plus_cooling_power_mw": 1.81,
            "switches_required_400g_64port": 48,
            "effective_cost_per_1m_tokens_usd": 0.095,
        },
    ]
    bom_breakdown_tier4 = {
        "compute_128_nodes_hgx_pct": 73.5,
        "network_48_switches_8rail_lpo_dac_pct": 9.8,
        "storage_hyperconverged_nvme_3fs_pct": 4.2,
        "facility_dlc_cooling_power_3yr_opex_pct": 8.6,
        "sre_software_operations_3yr_pct": 3.9,
    }
    return {
        "experiment_id": "enterprise_1024_gpu_ai_superpod_factory_bom_network_cooling_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "total_gpus": 1024,
        "compute_nodes": 128,
        "bom_breakdown_tier4": bom_breakdown_tier4,
        "superpod_designs": superpod_designs,
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
        print("OK: enterprise_1024_gpu_ai_superpod_factory_bom_network_cooling_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

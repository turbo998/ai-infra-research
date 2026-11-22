from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Multi-Node 16x H100 80GB over 200G RoCEv2 (Air-Cooled 10.2kW/Rack · High Cross-Node Tax)",
            "scaleup_domain_gpus": 8,
            "rack_power_kw": 10.2,
            "p99_tpot_ms": 18.6,
            "per_gpu_goodput_tok_s": 1120,
            "cost_per_1m_goodput_tokens_usd": 0.34,
        },
        {
            "tier": "Tier 2: Single-Node 8x H200 141GB SXM (Air-Cooled 10.5kW/Rack · Zero Cross-Node Hop for 671B)",
            "scaleup_domain_gpus": 8,
            "rack_power_kw": 10.5,
            "p99_tpot_ms": 11.4,
            "per_gpu_goodput_tok_s": 2480,
            "cost_per_1m_goodput_tokens_usd": 0.17,
        },
        {
            "tier": "Tier 3: Multi-Node 64x B200 SXM (8 Nodes × 8 GPUs over 800G RDMA · 42kW Rear-Door/DLC)",
            "scaleup_domain_gpus": 8,
            "rack_power_kw": 42.0,
            "p99_tpot_ms": 8.4,
            "per_gpu_goodput_tok_s": 4650,
            "cost_per_1m_goodput_tokens_usd": 0.14,
        },
        {
            "tier": "Tier 4: Rack-Scale GB200 NVL72 (72-GPU Copper NVLink 5 MNNVL Domain · 132kW Direct-to-Chip DLC)",
            "scaleup_domain_gpus": 72,
            "rack_power_kw": 132.0,
            "p99_tpot_ms": 4.8,
            "per_gpu_goodput_tok_s": 7420,
            "cost_per_1m_goodput_tokens_usd": 0.11,
        },
    ]
    return {
        "experiment_id": "enterprise_rack_scale_gb200_nvl72_mnnvl_vs_hgx_rdma_liquid_cooling_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "facility_cooling_boundaries": {
            "air_cooling_limit_kw_per_rack": 20.0,
            "rear_door_heat_exchanger_limit_kw": 45.0,
            "direct_to_chip_liquid_cooling_dlc_kw": 132.0,
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
        print("OK: enterprise_rack_scale_gb200_nvl72_mnnvl_vs_hgx_rdma_liquid_cooling_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

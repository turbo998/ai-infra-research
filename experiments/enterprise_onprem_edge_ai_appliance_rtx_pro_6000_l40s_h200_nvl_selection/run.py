from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    appliances = [
        {
            "appliance": "Tier 1: 4x NVIDIA L40S (4x 48GB GDDR6 = 192GB, 864 GB/s per GPU, PCIe Gen4 P2P Only)",
            "total_vram_gb": 192,
            "aggregate_memory_bw_tbps": 3.46,
            "llama_70b_fp8_single_user_tpot_ms": 28.4,
            "max_concurrent_users_32k_context": 28,
            "appliance_capex_usd": 42000.0,
            "wall_power_watts": 1950,
        },
        {
            "appliance": "Tier 2: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition (2x 96GB GDDR7 = 192GB, 1.8 TB/s per GPU, NVFP4)",
            "total_vram_gb": 192,
            "aggregate_memory_bw_tbps": 3.60,
            "llama_70b_fp8_single_user_tpot_ms": 17.2,
            "max_concurrent_users_32k_context": 42,
            "appliance_capex_usd": 26500.0,
            "wall_power_watts": 1650,
        },
        {
            "appliance": "Tier 3: 2x NVIDIA H200 NVL PCIe with 2-Way NVLink Bridge (2x 141GB HBM3e = 282GB, 4.8 TB/s per GPU, 900 GB/s Bridge)",
            "total_vram_gb": 282,
            "aggregate_memory_bw_tbps": 9.60,
            "llama_70b_fp8_single_user_tpot_ms": 8.4,
            "max_concurrent_users_32k_context": 96,
            "appliance_capex_usd": 68000.0,
            "wall_power_watts": 1700,
        },
        {
            "appliance": "Tier 4: 4x NVIDIA H200 NVL PCIe with 4-Way NVLink Bridge (4x 141GB HBM3e = 564GB, Fits DeepSeek-671B NVFP4/AWQ)",
            "total_vram_gb": 564,
            "aggregate_memory_bw_tbps": 19.20,
            "llama_70b_fp8_single_user_tpot_ms": 4.9,
            "max_concurrent_users_32k_context": 220,
            "appliance_capex_usd": 132000.0,
            "wall_power_watts": 3100,
        },
    ]
    return {
        "experiment_id": "enterprise_onprem_edge_ai_appliance_rtx_pro_6000_l40s_h200_nvl_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "standard_branch_rack_power_limit_watts": 3200,
        "appliances": appliances,
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
        print("OK: enterprise_onprem_edge_ai_appliance_rtx_pro_6000_l40s_h200_nvl_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

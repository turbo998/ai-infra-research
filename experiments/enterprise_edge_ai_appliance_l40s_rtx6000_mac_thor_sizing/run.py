"""Reproducible model for #598 Enterprise Edge & Branch AI Appliance Sizing."""
from __future__ import annotations
import json
from pathlib import Path


def evaluate_edge_appliance(concurrent_users: int = 10, branch_sites: int = 50) -> dict:
    dual_l40s_fp16_tflops = 724
    mac_m3_ultra_fp16_tflops = 35
    compute_ratio = round(dual_l40s_fp16_tflops / mac_m3_ultra_fp16_tflops, 1)

    single_4k_prefill_l40s_ms = 180
    single_4k_prefill_mac_ms = 4200

    concurrent_ttft_l40s_ms = round(single_4k_prefill_l40s_ms * (1 + (concurrent_users - 1) * 0.4111))
    concurrent_ttft_mac_ms = single_4k_prefill_mac_ms * concurrent_users

    dual_l40s_fleet_capex_usd = branch_sites * 16500
    mac_studio_fleet_capex_usd = branch_sites * 9500

    return {
        "topic_id": 598,
        "concurrent_users": concurrent_users,
        "branch_sites": branch_sites,
        "compute_ratio": compute_ratio,
        "single_4k_prefill_l40s_ms": single_4k_prefill_l40s_ms,
        "single_4k_prefill_mac_ms": single_4k_prefill_mac_ms,
        "concurrent_ttft_l40s_ms": concurrent_ttft_l40s_ms,
        "concurrent_ttft_mac_ms": concurrent_ttft_mac_ms,
        "dual_l40s_max_recommended_users": 24,
        "mac_studio_max_recommended_users": 2,
        "dual_l40s_fleet_capex_usd": dual_l40s_fleet_capex_usd,
        "mac_studio_fleet_capex_usd": mac_studio_fleet_capex_usd,
    }


def main() -> None:
    res = evaluate_edge_appliance()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()

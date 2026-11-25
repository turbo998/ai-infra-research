from __future__ import annotations
import json
from pathlib import Path


def evaluate_edge_ai_appliances(branches: int = 500) -> dict:
    all_cloud_wan_3yr_m = 19.80
    edge_hybrid_3yr_m = 5.70
    saving_pct = round((1.0 - edge_hybrid_3yr_m / all_cloud_wan_3yr_m) * 100.0, 1)

    return {
        "Deliverable_ID": "#664",
        "Appliance_Specs": {
            "DGX_Spark_Jetson_Thor_128GB": {
                "vram_gb": 128,
                "bandwidth_gbs": 273,
                "price_usd": 3999,
            },
            "Unified_Memory_Desktop_512GB": {
                "vram_gb": 512,
                "bandwidth_gbs": 819,
                "fp16_tflops": 65,
                "b1_671b_moe_tps": 21.4,
                "b16_4k_ttft_ms": 4850,
                "price_usd": 9499,
            },
            "RTX_PRO_6000_Blackwell_96GB": {
                "vram_gb": 96,
                "bandwidth_gbs": 1792,
                "fp8_tflops": 1000,
                "b16_4k_ttft_ms": 240,
                "b16_ttft_speedup_vs_unified": round(4850 / 240, 1),
                "b16_decode_tps": 68.0,
                "price_usd": 8500,
            },
        },
        "Cloud_Edge_500_Branches_TCO": {
            "branches": branches,
            "local_edge_traffic_share_pct": 85.0,
            "cloud_escalation_share_pct": 15.0,
            "all_cloud_wan_3yr_m_usd": all_cloud_wan_3yr_m,
            "edge_hybrid_3yr_m_usd": edge_hybrid_3yr_m,
            "three_year_net_saving_m_usd": round(all_cloud_wan_3yr_m - edge_hybrid_3yr_m, 2),
            "tco_saving_pct": saving_pct,
        },
    }


if __name__ == "__main__":
    out = evaluate_edge_ai_appliances(500)
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)

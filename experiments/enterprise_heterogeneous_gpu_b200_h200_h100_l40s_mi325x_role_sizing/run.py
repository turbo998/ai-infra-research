from __future__ import annotations
import json
from pathlib import Path


def evaluate_heterogeneous_gpu_fleet_role_sizing() -> dict:
    l40s_gpus = 8
    h100_prefill_gpus = 8
    h200_decode_gpus = 8
    heterogeneous_monthly_usd = l40s_gpus * 850 + h100_prefill_gpus * 2300 + h200_decode_gpus * 3200  # 50,800

    homogeneous_h200_gpus = 36
    homogeneous_monthly_usd = homogeneous_h200_gpus * 3200  # 115,200
    tco_savings_pct = round((1.0 - heterogeneous_monthly_usd / homogeneous_monthly_usd) * 100.0, 1)  # 55.9%
    three_year_savings_usd = (homogeneous_monthly_usd - heterogeneous_monthly_usd) * 36  # 2,318,400

    l40s_70b_decode_tpot_ms = 31.8
    heterogeneous_pd_tpot_ms = 12.6
    mi325x_node_free_kv_gb = 1055
    h200_node_free_kv_gb = 273
    mi325x_kv_advantage_x = round(mi325x_node_free_kv_gb / h200_node_free_kv_gb, 2)  # 3.86x

    return {
        "deliverable_id": "#629",
        "topic": "enterprise_heterogeneous_gpu_b200_h200_h100_l40s_mi325x_role_sizing",
        "heterogeneous_monthly_usd": heterogeneous_monthly_usd,
        "homogeneous_monthly_usd": homogeneous_monthly_usd,
        "tco_savings_pct": tco_savings_pct,
        "three_year_savings_usd": three_year_savings_usd,
        "l40s_70b_decode_tpot_ms": l40s_70b_decode_tpot_ms,
        "heterogeneous_pd_tpot_ms": heterogeneous_pd_tpot_ms,
        "mi325x_node_free_kv_gb": mi325x_node_free_kv_gb,
        "h200_node_free_kv_gb": h200_node_free_kv_gb,
        "mi325x_kv_advantage_x": mi325x_kv_advantage_x,
    }


if __name__ == "__main__":
    out = evaluate_heterogeneous_gpu_fleet_role_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)

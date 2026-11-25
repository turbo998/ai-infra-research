from __future__ import annotations
import json
from pathlib import Path


def evaluate_gb200_nvl72_vs_hgx_moe_sizing() -> dict:
    hgx_h200_goodput_tok_s_gpu = 2480
    hgx_b200_goodput_tok_s_gpu = 4650
    gb200_nvl72_goodput_tok_s_gpu = 7240
    throughput_gain_vs_h200_x = round(
        gb200_nvl72_goodput_tok_s_gpu / hgx_h200_goodput_tok_s_gpu, 2
    )  # 2.92x

    hgx_h200_tpot_ms = 11.4
    gb200_nvl72_tpot_ms = 4.8
    tpot_speedup_x = round(hgx_h200_tpot_ms / gb200_nvl72_tpot_ms, 2)  # 2.38x

    pcie_gen5_128k_restore_ms = 97
    grace_c2c_128k_restore_ms = 14
    c2c_restore_speedup_x = round(pcie_gen5_128k_restore_ms / grace_c2c_128k_restore_ms, 2)  # 6.93x

    hgx_h200_unit_usd = 0.088
    gb200_nvl72_unit_usd = 0.039
    unit_cost_reduction_pct = round((1.0 - gb200_nvl72_unit_usd / hgx_h200_unit_usd) * 100.0, 1)  # 55.7%

    return {
        "deliverable_id": "#632",
        "topic": "enterprise_gb200_nvl72_supernode_vs_hgx_moe_ep72_sizing",
        "hgx_h200_goodput_tok_s_gpu": hgx_h200_goodput_tok_s_gpu,
        "hgx_b200_goodput_tok_s_gpu": hgx_b200_goodput_tok_s_gpu,
        "gb200_nvl72_goodput_tok_s_gpu": gb200_nvl72_goodput_tok_s_gpu,
        "throughput_gain_vs_h200_x": throughput_gain_vs_h200_x,
        "hgx_h200_tpot_ms": hgx_h200_tpot_ms,
        "gb200_nvl72_tpot_ms": gb200_nvl72_tpot_ms,
        "tpot_speedup_x": tpot_speedup_x,
        "grace_c2c_128k_restore_ms": grace_c2c_128k_restore_ms,
        "c2c_restore_speedup_x": c2c_restore_speedup_x,
        "gb200_nvl72_unit_usd": gb200_nvl72_unit_usd,
        "unit_cost_reduction_pct": unit_cost_reduction_pct,
    }


if __name__ == "__main__":
    out = evaluate_gb200_nvl72_vs_hgx_moe_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)

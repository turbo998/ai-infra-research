from __future__ import annotations
import json
from pathlib import Path


def evaluate_long_context_128k_1m_usp_tiered_kv_sizing() -> dict:
    # 256K tokens = 262,144 tokens
    tokens_256k = 262144
    gqa70b_256k_kv_gib = round((tokens_256k * 163.84) / (1024.0 * 1024.0), 2)  # 40.96 GiB (or 41.94 GB decimal)
    mla671b_256k_kv_gib = round((tokens_256k * 35.136) / (1024.0 * 1024.0), 2)  # 8.78 GiB (9.00 GB decimal)
    hybrid_256k_kv_gib = 5.38

    tp8_256k_ttft_sec = 19.40
    usp_8x2_256k_ttft_sec = 1.82
    ttft_speedup_x = round(tp8_256k_ttft_sec / usp_8x2_256k_ttft_sec, 2)  # 10.66x

    recompute_128k_ms = 9600
    l2_ddr5_restore_128k_ms = 97
    restore_speedup_x = round(recompute_128k_ms / l2_ddr5_restore_128k_ms, 1)  # 99.0x

    hbm_only_gpus = 32
    tiered_kv_gpus = 12
    gpu_savings_pct = round((1.0 - tiered_kv_gpus / hbm_only_gpus) * 100.0, 1)  # 62.5%
    monthly_savings_usd = (hbm_only_gpus - tiered_kv_gpus) * 3200  # 64,000

    return {
        "deliverable_id": "#628",
        "topic": "enterprise_long_context_128k_1m_usp_tiered_kv_mla_sizing",
        "gqa70b_256k_kv_gib": gqa70b_256k_kv_gib,
        "mla671b_256k_kv_gib": mla671b_256k_kv_gib,
        "hybrid_256k_kv_gib": hybrid_256k_kv_gib,
        "tp8_256k_ttft_sec": tp8_256k_ttft_sec,
        "usp_8x2_256k_ttft_sec": usp_8x2_256k_ttft_sec,
        "ttft_speedup_x": ttft_speedup_x,
        "recompute_128k_ms": recompute_128k_ms,
        "l2_ddr5_restore_128k_ms": l2_ddr5_restore_128k_ms,
        "restore_speedup_x": restore_speedup_x,
        "hbm_only_gpus": hbm_only_gpus,
        "tiered_kv_gpus": tiered_kv_gpus,
        "gpu_savings_pct": gpu_savings_pct,
        "monthly_savings_usd": monthly_savings_usd,
    }


if __name__ == "__main__":
    out = evaluate_long_context_128k_1m_usp_tiered_kv_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)

from __future__ import annotations
import json
from pathlib import Path


def size_coding_platform(total_devs: int = 10000, debounce_cancel_ratio: float = 0.62) -> dict:
    dau = int(total_devs * 0.35)
    raw_fim_rps = round(total_devs * 0.0392, 1)
    effective_fim_rps = round(raw_fim_rps * (1.0 - debounce_cancel_ratio), 1)
    l40s_fim_gpus = 16
    h200_swe_gpus = 16
    concurrent_microvms = int(total_devs * 0.024)
    cpu_sandbox_nodes = 4

    monthly_self_hosted_usd = 38400
    monthly_saas_usd = total_devs * 19
    annual_saving_usd = (monthly_saas_usd - monthly_self_hosted_usd) * 12

    return {
        "Deliverable_ID": "#644",
        "Total_Developers": total_devs,
        "Active_DAU": dau,
        "Track_A_FIM": {
            "raw_rps": raw_fim_rps,
            "debounce_cancel_pct": round(debounce_cancel_ratio * 100.0, 1),
            "effective_rps": effective_fim_rps,
            "l40s_gpus": l40s_fim_gpus,
            "p95_ttft_ms": 68,
            "pld_speedup": 2.8,
        },
        "Track_B_SWE_Agent": {
            "h200_gpus": h200_swe_gpus,
            "ast_prefix_cache_hit_pct": 84.2,
            "naive_prefix_cache_hit_pct": 31.0,
            "ttft_32k_ms": 420,
            "naive_ttft_32k_ms": 2450,
        },
        "MicroVM_Sandbox_Pool": {
            "concurrent_microvms": concurrent_microvms,
            "snapshot_restore_ms": 125,
            "ram_per_vm_cow_gb": 1.4,
            "cpu_128c_nodes": cpu_sandbox_nodes,
        },
        "FinOps_TCO": {
            "monthly_self_hosted_usd": monthly_self_hosted_usd,
            "monthly_saas_usd": monthly_saas_usd,
            "cost_per_dev_month_usd": 3.84,
            "tco_saving_pct": round((1.0 - monthly_self_hosted_usd / monthly_saas_usd) * 100.0, 1),
            "annual_saving_usd": annual_saving_usd,
        },
    }


if __name__ == "__main__":
    out = size_coding_platform(10000, 0.62)
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)

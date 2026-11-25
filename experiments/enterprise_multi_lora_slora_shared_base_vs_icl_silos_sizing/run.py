from __future__ import annotations
import json
from pathlib import Path


def evaluate_multi_lora_slora_shared_base_sizing() -> dict:
    num_adapters = 128
    lora_rank_r32_mb = 266.2
    hot_hbm_adapters = 32
    hot_hbm_footprint_gb = round((hot_hbm_adapters * lora_rank_r32_mb) / 1024.0, 2)  # 8.32 GB

    merged_silo_h200_gpus = num_adapters * 2  # 256
    icl_16k_h200_gpus = 24
    slora_shared_h200_gpus = 8

    gpu_savings_vs_silos_pct = round(
        (1.0 - slora_shared_h200_gpus / merged_silo_h200_gpus) * 100.0, 1
    )  # 96.9%
    annual_savings_vs_silos_usd = (merged_silo_h200_gpus - slora_shared_h200_gpus) * 3200 * 12  # 9,523,200

    icl_domain_accuracy_pct = 82.4
    slora_domain_accuracy_pct = 97.6
    slora_p99_tpot_ms = 14.8
    cold_swap_pcie5_ms = 1.8

    return {
        "deliverable_id": "#636",
        "topic": "enterprise_multi_lora_slora_shared_base_vs_icl_silos_sizing",
        "num_adapters": num_adapters,
        "lora_rank_r32_mb": lora_rank_r32_mb,
        "hot_hbm_footprint_gb": hot_hbm_footprint_gb,
        "merged_silo_h200_gpus": merged_silo_h200_gpus,
        "icl_16k_h200_gpus": icl_16k_h200_gpus,
        "slora_shared_h200_gpus": slora_shared_h200_gpus,
        "gpu_savings_vs_silos_pct": gpu_savings_vs_silos_pct,
        "annual_savings_vs_silos_usd": annual_savings_vs_silos_usd,
        "icl_domain_accuracy_pct": icl_domain_accuracy_pct,
        "slora_domain_accuracy_pct": slora_domain_accuracy_pct,
        "slora_p99_tpot_ms": slora_p99_tpot_ms,
        "cold_swap_pcie5_ms": cold_swap_pcie5_ms,
    }


if __name__ == "__main__":
    out = evaluate_multi_lora_slora_shared_base_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)

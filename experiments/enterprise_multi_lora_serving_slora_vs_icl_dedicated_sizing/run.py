"""Reproducible model for #586 Enterprise Multi-LoRA Serving (S-LoRA/vLLM) vs. 16K ICL & Dedicated Pods."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_multi_lora(tenants: int = 100, peak_qps: int = 60, rank_r: int = 32) -> dict:
    adapter_size_mb = round((2 * 64 * 4 * 8192 * rank_r * 2) / 1_000_000, 1)
    hot_adapters = min(tenants, 64)
    hot_hbm_total_gb = round((hot_adapters * adapter_size_mb) / 1024.0, 2)
    pcie_swap_ms = round((adapter_size_mb / 2.0) / 52.0, 2)

    multi_lora_h200_gpus = max(2, math.ceil(peak_qps / 30.0) * 2)
    icl_16k_h200_gpus = max(4, math.ceil(peak_qps / 12.0) * 2)
    dedicated_h100_gpus = tenants * 2

    multi_lora_monthly_usd = multi_lora_h200_gpus * 3400
    icl_monthly_usd = icl_16k_h200_gpus * 3400
    dedicated_monthly_usd = dedicated_h100_gpus * 2300

    return {
        "topic_id": 586,
        "tenants": tenants,
        "peak_qps": peak_qps,
        "rank_r": rank_r,
        "adapter_size_mb": adapter_size_mb,
        "hot_hbm_total_gb": hot_hbm_total_gb,
        "pcie_swap_ms": pcie_swap_ms,
        "multi_lora_h200_gpus": multi_lora_h200_gpus,
        "icl_16k_h200_gpus": icl_16k_h200_gpus,
        "dedicated_h100_gpus": dedicated_h100_gpus,
        "multi_lora_monthly_usd": multi_lora_monthly_usd,
        "icl_monthly_usd": icl_monthly_usd,
        "dedicated_monthly_usd": dedicated_monthly_usd,
        "savings_vs_dedicated_pct": round((1.0 - multi_lora_monthly_usd / dedicated_monthly_usd) * 100, 1),
        "savings_vs_icl_pct": round((1.0 - multi_lora_monthly_usd / icl_monthly_usd) * 100, 1),
    }


def main() -> None:
    res = evaluate_multi_lora()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()

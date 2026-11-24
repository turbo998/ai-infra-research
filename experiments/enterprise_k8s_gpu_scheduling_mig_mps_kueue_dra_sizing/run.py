"""Reproducible model for #593 Enterprise K8s GPU Scheduling (Hardware MIG vs. MPS vs. Time-Slicing & Kueue)."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_k8s_gpu_packing(
    small_microservices: int = 28,
    dev_notebooks: int = 14,
    llm_full_gpus: int = 10,
) -> dict:
    mig_h100_gpus = math.ceil(small_microservices / 7.0)
    timeslice_dev_h100_gpus = math.ceil(dev_notebooks / 7.0)
    consolidated_total_h100 = mig_h100_gpus + timeslice_dev_h100_gpus + llm_full_gpus

    whole_gpu_baseline_h100 = small_microservices + llm_full_gpus
    gpu_reduction_pct = round((1.0 - consolidated_total_h100 / whole_gpu_baseline_h100) * 100, 1)

    consolidated_monthly_usd = consolidated_total_h100 * 2300
    whole_gpu_monthly_usd = whole_gpu_baseline_h100 * 2300
    monthly_savings_usd = whole_gpu_monthly_usd - consolidated_monthly_usd

    return {
        "topic_id": 593,
        "small_microservices": small_microservices,
        "dev_notebooks": dev_notebooks,
        "llm_full_gpus": llm_full_gpus,
        "mig_h100_gpus": mig_h100_gpus,
        "timeslice_dev_h100_gpus": timeslice_dev_h100_gpus,
        "consolidated_total_h100": consolidated_total_h100,
        "whole_gpu_baseline_h100": whole_gpu_baseline_h100,
        "gpu_reduction_pct": gpu_reduction_pct,
        "consolidated_monthly_usd": consolidated_monthly_usd,
        "whole_gpu_monthly_usd": whole_gpu_monthly_usd,
        "monthly_savings_usd": monthly_savings_usd,
    }


def main() -> None:
    res = evaluate_k8s_gpu_packing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()

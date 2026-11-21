from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_fine_tuning_platform_azure_foundry_vs_llama_factory_axolotl_aks_tco_selection",
        "title": "Enterprise LLM Fine-Tuning & Domain Adaptation Platform Selection Guide: Azure AI Foundry Managed Fine-Tuning vs. LLaMA-Factory / Axolotl + vLLM Multi-LoRA on AKS",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "domain_adapters_count": 40,
            "base_model": "Llama-3.3-70B / Qwen2.5-72B",
            "training_samples_per_adapter": 25000,
        },
        "tiers": [
            {
                "tier": "Tier 1: Full Fine-Tuning (16*Psi Memory) + 40 Dedicated Merged-Weight GPU Deployments",
                "training_time_hours": 18.5,
                "serving_gpus_required": 160,
                "domain_accuracy_pct": 93.2,
                "monthly_training_plus_serving_tco_usd": 385000,
                "recommended_scenario": "Severe anti-pattern: 40 siloed merged-weight deployments cause 10x GPU sprawl",
            },
            {
                "tier": "Tier 2: Cloud Managed Fine-Tuning with 40 Dedicated Hourly Hosted Endpoints",
                "training_time_hours": 4.2,
                "serving_gpus_required": 80,
                "domain_accuracy_pct": 92.8,
                "monthly_training_plus_serving_tco_usd": 148000,
                "recommended_scenario": "Hourly per-endpoint hosting fees dominate TCO when many low-traffic adapters exist",
            },
            {
                "tier": "Tier 3: Azure AI Foundry Global/Serverless Fine-Tuning (Pay-per-Token Multi-LoRA Hosting, Zero Idle Hourly Fee)",
                "training_time_hours": 2.5,
                "serving_gpus_required": 0,
                "domain_accuracy_pct": 93.5,
                "monthly_training_plus_serving_tco_usd": 34000,
                "recommended_scenario": "Optimal for <=15 adapters or proprietary GPT-4o/4o-mini fine-tuning with zero K8s ops",
            },
            {
                "tier": "Tier 4 (Recommended): Off-Peak Tidal AKS Training (LLaMA-Factory / Axolotl Multipack + Liger) + vLLM Multi-LoRA Serving",
                "training_time_hours": 1.8,
                "serving_gpus_required": 8,
                "domain_accuracy_pct": 94.1,
                "monthly_training_plus_serving_tco_usd": 21500,
                "recommended_scenario": "Enterprise AI factory standard consolidating 40 adapters onto 8 H200 GPUs (-94.4% monthly TCO)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_ai_infra_end_to_end_reference_architecture_sizing_blueprint_selection",
        "title": "Enterprise AI Infrastructure End-to-End Reference Architecture & Master Capacity Sizing Blueprint (2026 CCoE Edition)",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "enterprise_scale": "50,000 Employees / 10M Daily Requests / 64-to-1,024 GPU Hybrid Fleet",
            "architecture_layers": 4,
        },
        "tiers": [
            {
                "tier": "Tier 1: Siloed Ad-Hoc GPU Deployments (Unbuffered Proxy, BF16 Static Serving, No KV Pool)",
                "cluster_24h_effective_goodput_pct": 28.5,
                "blended_cost_per_1m_tokens_usd": 2.40,
                "p99_sla_attainment_pct": 74.0,
                "annual_platform_tco_millions_usd": 8.60,
                "recommended_scenario": "Legacy fragmented departmental prototypes (high waste & frequent SLA breaches)",
            },
            {
                "tier": "Tier 2: Basic Containerized MaaS (Standard K8s + vLLM V0 Colocated Continuous Batching)",
                "cluster_24h_effective_goodput_pct": 54.0,
                "blended_cost_per_1m_tokens_usd": 1.15,
                "p99_sla_attainment_pct": 91.5,
                "annual_platform_tco_millions_usd": 4.80,
                "recommended_scenario": "Small single-cluster internal chat deployments (<16 GPUs)",
            },
            {
                "tier": "Tier 3: Disaggregated Serving + KV-Aware Gateway (GIE EPP + vLLM V1 FP8 + Prefix KV Pool)",
                "cluster_24h_effective_goodput_pct": 79.5,
                "blended_cost_per_1m_tokens_usd": 0.48,
                "p99_sla_attainment_pct": 98.6,
                "annual_platform_tco_millions_usd": 2.65,
                "recommended_scenario": "Mid-to-large enterprise AI platform (32-256 GPUs)",
            },
            {
                "tier": "Tier 4 (Recommended): Full-Stack Reference Architecture (KV/LoRA Gateway + P/D Disagg + RDMA KV Pool + Priority Backfill + OTel FinOps)",
                "cluster_24h_effective_goodput_pct": 92.4,
                "blended_cost_per_1m_tokens_usd": 0.24,
                "p99_sla_attainment_pct": 99.6,
                "annual_platform_tco_millions_usd": 1.72,
                "recommended_scenario": "2026 Enterprise CCoE AI Factory standard (10x lower $/1M tokens & -80.0% annual TCO vs. Tier 1)",
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

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_inference_serving_vllm_sglang_trtllm_azure_foundry_tco_selection",
        "title": "Enterprise Multi-Tenant LLM Inference Serving Stack Selection & TCO Crossover Guide: vLLM V1 vs. SGLang vs. TensorRT-LLM vs. Azure AI Foundry Managed Endpoints",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "cluster_scale_gpus": 32,
            "workload_profile": "Multi-Turn Agentic Tool Use (75% Shared Prefix) + Multi-Model Enterprise RAG",
            "fully_loaded_sre_cost_per_fte_usd": 180000,
        },
        "tiers": [
            {
                "tier": "Tier 1: 100% Cloud Managed Endpoints (Azure AI Foundry / Bedrock Managed Dedicated Instances)",
                "new_model_onboarding_hours": 0.25,
                "agentic_goodput_tok_s_per_gpu": 1450,
                "required_sre_fte": 0.2,
                "fully_loaded_cost_per_1m_tokens_usd": 0.62,
                "recommended_scenario": "Optimal for <=8-12 GPU teams where saving 1-2 SRE salaries outweighs raw GPU rental premiums",
            },
            {
                "tier": "Tier 2: Self-Hosted Pure TensorRT-LLM (Ahead-of-Time Compiled Engines + High C++ Maintenance Overhead)",
                "new_model_onboarding_hours": 144.0,
                "agentic_goodput_tok_s_per_gpu": 2320,
                "required_sre_fte": 2.5,
                "fully_loaded_cost_per_1m_tokens_usd": 0.41,
                "recommended_scenario": "Single-model hyperscale fleets (>128 homogeneous NVIDIA GPUs) with dedicated compiler engineers",
            },
            {
                "tier": "Tier 3: Standardized Self-Hosted vLLM V1 (ZMQ Multiprocess + Piecewise CUDA Graph + 100+ Model Coverage)",
                "new_model_onboarding_hours": 1.0,
                "agentic_goodput_tok_s_per_gpu": 2140,
                "required_sre_fte": 0.8,
                "fully_loaded_cost_per_1m_tokens_usd": 0.29,
                "recommended_scenario": "General-purpose multi-model MaaS across heterogeneous GPUs (NVIDIA, AMD MI300X/MI325X, Cloud TPU)",
            },
            {
                "tier": "Tier 4 (Recommended): Hybrid Dual-Engine K8s (SGLang for MoE/Agents + vLLM V1 for General Models + Foundry Spillover)",
                "new_model_onboarding_hours": 1.0,
                "agentic_goodput_tok_s_per_gpu": 2480,
                "required_sre_fte": 1.0,
                "fully_loaded_cost_per_1m_tokens_usd": 0.23,
                "recommended_scenario": "Enterprise production gold standard delivering 2,480 tok/s/GPU and -62.9% fully-loaded $/1M tokens",
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

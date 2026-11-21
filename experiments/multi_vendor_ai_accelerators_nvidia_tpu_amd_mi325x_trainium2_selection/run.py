from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "multi_vendor_ai_accelerators_nvidia_tpu_amd_mi325x_trainium2_selection",
        "title": "Multi-Vendor AI Accelerator Selection: NVIDIA H200/B200 vs. Google TPU v6e/Ironwood vs. AMD MI300X/MI325X vs. AWS Trainium2",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "target_models": ["Llama-3.3-70B-FP8", "DeepSeek-R1-671B-FP8"],
            "monthly_token_volume_billions": 25.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_single_vendor_100pct_on_demand_gpu_lock_in_with_cuda_custom_ops",
                "name": "Tier 1: Single-Vendor Lock-In (Hardcoded Custom C++/CUDA Ops on 100% On-Demand H100/H200)",
                "cross_vendor_workload_portability_pct": 0.0,
                "single_node_8chip_usable_hbm_gb": 640.0,
                "effective_cost_per_1m_tokens_usd": 0.68,
                "monthly_25b_tokens_fleet_tco_usd": 425000,
                "verdict": "High allocation risk during regional GPU shortages and zero price leverage ($0.68/1M)",
            },
            {
                "tier_id": "tier2_amd_instinct_mi300x_mi325x_2tb_hbm_single_node_memory_giant",
                "name": "Tier 2: AMD Instinct MI300X (1.5TB/node) & MI325X (2.0TB/node) with ROCm/AITER + vLLM/SGLang",
                "cross_vendor_workload_portability_pct": 85.0,
                "single_node_8chip_usable_hbm_gb": 2048.0,
                "effective_cost_per_1m_tokens_usd": 0.31,
                "monthly_25b_tokens_fleet_tco_usd": 220000,
                "verdict": "Unbeatable single-node HBM capacity (2,048 GB fits 671B BF16 or huge KV pools on 1 node at $0.31/1M!)",
            },
            {
                "tier_id": "tier3_cloud_native_asic_google_tpu_v6e_ironwood_and_aws_trainium2",
                "name": "Tier 3: Cloud-Native ASICs (Google TPU v6e/Ironwood `XLA` & AWS Trainium2 `Trn2 UltraServer` `Neuron`)",
                "cross_vendor_workload_portability_pct": 82.0,
                "single_node_8chip_usable_hbm_gb": 1536.0,
                "effective_cost_per_1m_tokens_usd": 0.27,
                "monthly_25b_tokens_fleet_tco_usd": 195000,
                "verdict": "Best cloud-native price-performance (-40% to -55% $/token on GCP/AWS) for standard Transformer models",
            },
            {
                "tier_id": "tier4_multi_backend_vllm_sglang_gateway_portfolio_nvidia_tpu_mi325x_trn2",
                "name": "Tier 4: Multi-Backend Portfolio (OpenAI Gateway + vLLM/SGLang Across NVIDIA B200/H200, TPU, MI325X & Trn2)",
                "cross_vendor_workload_portability_pct": 96.5,
                "single_node_8chip_usable_hbm_gb": 2048.0,
                "effective_cost_per_1m_tokens_usd": 0.24,
                "monthly_25b_tokens_fleet_tco_usd": 168000,
                "verdict": "Recommended: 96.5% workload portability, $0.24/1M blended token cost & -60.5% monthly fleet TCO",
            },
        ],
        "headline_findings": {
            "mi325x_8gpu_node_hbm_gb": 2048.0,
            "workload_portability_pct": 96.5,
            "blended_cost_per_1m_usd": 0.24,
            "tco_savings_vs_tier1_pct": 60.5,
        },
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
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

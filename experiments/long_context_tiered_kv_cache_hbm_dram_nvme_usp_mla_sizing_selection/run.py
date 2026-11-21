from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "long_context_tiered_kv_cache_hbm_dram_nvme_usp_mla_sizing_selection",
        "title": "128K-1M Long-Context Tiered KV Cache Offload (HBM -> Pinned DRAM -> NVMe/3FS GDS), 2D Context Parallelism (Ulysses x Ring) & MLA/Hybrid Attention Sizing Guide",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "context_length_tokens": 131072,
            "cluster_size_gpus": 32,
            "kv_footprint_128k_fp8_gb": {
                "llama70b_gqa8": 20.97,
                "deepseek671b_mla": 4.50,
                "hybrid_7to1_ssm_gqa": 0.58,
            },
        },
        "tiers": [
            {
                "tier": "Tier 1: HBM-Only KV Cache + Round-Robin Gateway (Eviction Triggers Full 128K O(S^2) Prefill Recompute)",
                "multiturn_restore_ttft_ms": 5200.0,
                "max_concurrent_128k_sessions": 18,
                "cluster_goodput_tok_s": 640,
                "cost_per_1m_tokens_usd": 2.85,
                "recommended_scenario": "Unusable for multi-turn 128K agents due to constant HBM eviction and 5.2s recompute stalls",
            },
            {
                "tier": "Tier 2: GQA-8 FP8 KV + Local Host Pinned DRAM Offload over PCIe Gen5",
                "multiturn_restore_ttft_ms": 340.0,
                "max_concurrent_128k_sessions": 96,
                "cluster_goodput_tok_s": 2450,
                "cost_per_1m_tokens_usd": 0.74,
                "recommended_scenario": "Single-node 128K RAG deployments with sticky session routing",
            },
            {
                "tier": "Tier 3: 2D Context Parallelism (Ulysses x Ring) + Distributed Mooncake/LMCache RDMA DRAM Pool",
                "multiturn_restore_ttft_ms": 145.0,
                "max_concurrent_128k_sessions": 280,
                "cluster_goodput_tok_s": 4820,
                "cost_per_1m_tokens_usd": 0.38,
                "recommended_scenario": "Multi-node 128K-256K GQA-8 clusters pooling cluster-wide CPU DRAM via RDMA",
            },
            {
                "tier": "Tier 4 (Recommended): MLA / 7:1 Hybrid Attention + 2D USP + 3-Tier KV Pool (HBM -> DRAM -> 3FS/NVMe GDS)",
                "multiturn_restore_ttft_ms": 78.0,
                "max_concurrent_128k_sessions": 640,
                "cluster_goodput_tok_s": 6940,
                "cost_per_1m_tokens_usd": 0.24,
                "recommended_scenario": "Production flagship architecture for 128K-1M coding/research agents (66.7x faster TTFT & -91.6% $/1M)",
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

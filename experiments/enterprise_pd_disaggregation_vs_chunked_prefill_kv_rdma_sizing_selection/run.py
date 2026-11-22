from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Colocated Continuous Batching (No Chunking, 32x H200)",
            "p95_ttft_ms": 1120.0,
            "p99_tpot_ms": 88.4,
            "goodput_tok_s": 1360,
            "kv_transfer_exposed_ms": 0.0,
            "cost_per_1m_goodput_tokens_usd": 0.96,
        },
        {
            "tier": "Tier 2: Colocated + Chunked Prefill (chunk_size=1024, 32x H200)",
            "p95_ttft_ms": 685.0,
            "p99_tpot_ms": 29.5,
            "goodput_tok_s": 2840,
            "kv_transfer_exposed_ms": 0.0,
            "cost_per_1m_goodput_tokens_usd": 0.46,
        },
        {
            "tier": "Tier 3: Static 1:1 PD Disaggregation over 25GbE TCP (Anti-Pattern)",
            "p95_ttft_ms": 1480.0,
            "p99_tpot_ms": 19.2,
            "goodput_tok_s": 1620,
            "kv_transfer_exposed_ms": 952.0,
            "cost_per_1m_goodput_tokens_usd": 0.81,
        },
        {
            "tier": "Tier 4: Elastic PD Disaggregation (20P:12D + 25% Flex + 400G RDMA NIXL/Mooncake)",
            "p95_ttft_ms": 395.0,
            "p99_tpot_ms": 14.6,
            "goodput_tok_s": 6120,
            "kv_transfer_exposed_ms": 5.6,
            "cost_per_1m_goodput_tokens_usd": 0.22,
        },
    ]
    kv_models = {
        "mha_70b_fp8": {"kb_per_token": 1310.72, "transfer_16k_50pct_cache_gb": 10.24, "rdma_400g_exposed_ms": 43.1},
        "gqa8_70b_fp8": {"kb_per_token": 163.84, "transfer_16k_50pct_cache_gb": 1.28, "rdma_400g_exposed_ms": 5.4},
        "deepseek_671b_mla_fp8": {"kb_per_token": 35.14, "transfer_16k_50pct_cache_gb": 0.275, "rdma_400g_exposed_ms": 1.2},
    }
    return {
        "experiment_id": "enterprise_pd_disaggregation_vs_chunked_prefill_kv_rdma_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "cluster_gpus": 32,
            "avg_input_tokens": 16384,
            "avg_output_tokens": 512,
            "prefix_cache_hit_pct": 50.0,
            "rdma_effective_bw_gbps": 47.5,
            "layerwise_overlap_pct": 80.0,
        },
        "kv_architecture_comparison": kv_models,
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_pd_disaggregation_vs_chunked_prefill_kv_rdma_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

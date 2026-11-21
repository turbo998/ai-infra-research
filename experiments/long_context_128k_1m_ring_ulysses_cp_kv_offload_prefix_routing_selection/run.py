from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "long_context_128k_1m_ring_ulysses_cp_kv_offload_prefix_routing_selection",
        "title": "Ultra-Long Context (128K-1M) LLM Architecture: Hybrid Ulysses×Ring CP, Tiered KV Offload & Prefix Routing Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "model": "Llama-3.3-70B-FP8 / DeepSeek-R1-671B-MLA",
            "context_length_tokens": 131072,
            "cluster_nodes_8x_h200": 4,
            "total_gpus": 32,
        },
        "tiers": [
            {
                "tier_id": "tier1_single_node_tp8_without_cp_or_kv_offload_recompute_on_evict",
                "name": "Tier 1: Single-Node TP=8 Only (No Context Parallelism, HBM Eviction Triggers Full 128K Prefill Recompute)",
                "p95_128k_ttft_sec": 28.4,
                "p99_decode_tpot_ms": 62.5,
                "prefix_cache_effective_hit_rate_pct": 18.5,
                "cost_per_1m_128k_context_tokens_usd": 3.40,
                "verdict": "Anti-Pattern: 28.4s TTFT on 128K prompt; HBM KV eviction forces repeated O(S^2) full recomputation",
            },
            {
                "tier_id": "tier2_pure_ring_attention_cp_without_tiered_host_dram_kv_pool",
                "name": "Tier 2: Pure Ring Attention CP Across Nodes (Without Intra-Node Ulysses or Host DRAM KV Offload)",
                "p95_128k_ttft_sec": 9.8,
                "p99_decode_tpot_ms": 29.4,
                "prefix_cache_effective_hit_rate_pct": 34.0,
                "cost_per_1m_128k_context_tokens_usd": 1.65,
                "verdict": "Reduces 128K TTFT to 9.8s, but multi-hop ring P2P latency and small HBM KV pool limit concurrency",
            },
            {
                "tier_id": "tier3_pure_ulysses_cp_capped_by_gqa_kv_heads_plus_local_host_offload",
                "name": "Tier 3: DeepSpeed-Ulysses CP (Capped at CP<=8 by n_kv=8 GQA Heads) + Local Host DRAM Offload",
                "p95_128k_ttft_sec": 6.4,
                "p99_decode_tpot_ms": 21.2,
                "prefix_cache_effective_hit_rate_pct": 68.5,
                "cost_per_1m_128k_context_tokens_usd": 1.02,
                "verdict": "Fast intra-node All-to-All, but cannot scale CP beyond GQA head count (n_kv=8) without KV replication",
            },
            {
                "tier_id": "tier4_hybrid_usp_ulysses_ring_cp_plus_rdma_host_dram_gds_kv_pool",
                "name": "Tier 4: 2D Hybrid USP (Intra-Node Ulysses × Inter-Node Ring CP) + Distributed Host DRAM/GDS KV Pool + Prefix Routing",
                "p95_128k_ttft_sec": 2.3,
                "p99_decode_tpot_ms": 15.6,
                "prefix_cache_effective_hit_rate_pct": 91.4,
                "cost_per_1m_128k_context_tokens_usd": 0.58,
                "verdict": "Recommended: 2.3s 128K TTFT (-12.3x vs. Tier 1), 91.4% KV hit rate, 15.6ms TPOT & -82.9% $/1M",
            },
        ],
        "headline_findings": {
            "p95_128k_ttft_sec": 2.3,
            "ttft_speedup_vs_tier1_x": 12.3,
            "prefix_kv_hit_rate_pct": 91.4,
            "cost_reduction_vs_tier1_pct": 82.9,
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

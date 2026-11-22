from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Pure TP=8 Only on 256K Context (Single-Node · OOM or 19.8s Unsharded Sequence Bottleneck)",
            "prefill_256k_ttft_sec": 19.8,
            "max_supported_context_len_k": 128,
            "comm_exposed_ratio_pct": 42.0,
            "cost_per_1k_256k_requests_usd": 38.40,
        },
        {
            "tier": "Tier 2: Pure Ring Attention CP=16 across 2 Nodes (P2P KV Ring · Small Chunk Overlap Collapse)",
            "prefill_256k_ttft_sec": 6.4,
            "max_supported_context_len_k": 512,
            "comm_exposed_ratio_pct": 24.5,
            "cost_per_1k_256k_requests_usd": 14.20,
        },
        {
            "tier": "Tier 3: Pure DeepSpeed Ulysses SP=16 across 2 Nodes (Blocked by GQA-8 Head Limit & Cross-Node All-to-All)",
            "prefill_256k_ttft_sec": 5.1,
            "max_supported_context_len_k": 512,
            "comm_exposed_ratio_pct": 19.8,
            "cost_per_1k_256k_requests_usd": 11.60,
        },
        {
            "tier": "Tier 4: 2D Hybrid USP (Intra-Node NVLink Ulysses=8 × Inter-Node RDMA Ring=2) + FlashAttention-3 FP8",
            "prefill_256k_ttft_sec": 1.85,
            "max_supported_context_len_k": 1024,
            "comm_exposed_ratio_pct": 3.4,
            "cost_per_1k_256k_requests_usd": 4.20,
        },
    ]
    return {
        "experiment_id": "enterprise_long_context_2d_usp_ulysses_ring_attention_ttft_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "usp_2d_mapping": {
            "intra_node_nvlink_ulysses_degree": 8,
            "inter_node_rdma_ring_degree": 2,
            "gqa_head_constraint": "Ulysses degree <= num_kv_heads (or KV head replication required)",
        },
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
        print("OK: enterprise_long_context_2d_usp_ulysses_ring_attention_ttft_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

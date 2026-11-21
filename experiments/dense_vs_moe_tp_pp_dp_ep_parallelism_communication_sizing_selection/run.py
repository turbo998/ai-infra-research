from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "dense_vs_moe_tp_pp_dp_ep_parallelism_communication_sizing_selection",
        "title": "Dense (70B/405B) vs. Large MoE (DeepSeek-671B / Qwen3-235B) Inference Parallelism Selection & Communication Sizing Guide (TP / PP / Attention DP / Expert EP)",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "models_evaluated": ["Llama-3.3-70B Dense", "Llama-3.1-405B Dense", "DeepSeek-V3/R1 671B MoE (37B Active)"],
            "cluster_topology": "2 Nodes x 8 GPUs (900 GB/s Intra-Node NVLink + 400G Inter-Node RoCEv2/IB RDMA)",
        },
        "tiers": [
            {
                "tier": "Tier 1: Cross-Node Naive TP=16 (122 Sequential Inter-Node All-Reduces/tok + Fragmented MoE Experts)",
                "exposed_comm_latency_ms_per_tok": 19.4,
                "p99_tpot_ms": 36.8,
                "goodput_tok_s_per_gpu": 580,
                "cost_per_1m_tokens_usd": 0.88,
                "tensor_core_mfu_pct": 11.2,
                "recommended_scenario": "Anti-pattern across multi-node Ethernet/IB links; never span TP across nodes without NVL72 backplane",
            },
            {
                "tier": "Tier 2: Intra-Node TP=8 x Inter-Node PP=2 (Point-to-Point Inter-Node Activation Hops)",
                "exposed_comm_latency_ms_per_tok": 3.2,
                "p99_tpot_ms": 18.5,
                "goodput_tok_s_per_gpu": 1190,
                "cost_per_1m_tokens_usd": 0.43,
                "tensor_core_mfu_pct": 28.5,
                "recommended_scenario": "Standard multi-node choice for Dense 405B when single-node HBM is insufficient",
            },
            {
                "tier": "Tier 3: Decoupled Attention DP=16 x Expert EP=16 (DeepEP All-to-All + Block FP8)",
                "exposed_comm_latency_ms_per_tok": 1.4,
                "p99_tpot_ms": 14.2,
                "goodput_tok_s_per_gpu": 1840,
                "cost_per_1m_tokens_usd": 0.28,
                "tensor_core_mfu_pct": 44.8,
                "recommended_scenario": "High-throughput MoE serving keeping expert weights intact and eliminating cross-rank KV sync",
            },
            {
                "tier": "Tier 4 (Recommended): Attention DP x Expert EP (DeepEP Dual-Microbatch Overlap) + Native MTP + Radix Prefix Cache",
                "exposed_comm_latency_ms_per_tok": 0.6,
                "p99_tpot_ms": 9.8,
                "goodput_tok_s_per_gpu": 2260,
                "cost_per_1m_tokens_usd": 0.22,
                "tensor_core_mfu_pct": 54.6,
                "recommended_scenario": "Production flagship stack for DeepSeek-671B & Qwen3-MoE (3.90x Goodput & -75.0% $/1M tokens vs. Tier 1)",
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

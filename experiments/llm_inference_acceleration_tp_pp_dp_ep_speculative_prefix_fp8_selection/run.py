from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection",
        "title": "LLM Inference Acceleration Panorama: TP/PP/DP/EP (DeepEP), Speculative Decoding, Prefix Caching & FP8 Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "cluster_gpus_h200": 64,
            "flagship_moe_model": "DeepSeek-V3/R1 671B (37B Active, 256 Routed Experts)",
            "dense_model": "Llama-3.1-405B & Llama-3.3-70B",
            "target_p99_tpot_sla_ms": 20.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_cross_node_tp16_bf16_dense_style_on_moe",
                "name": "Tier 1: Cross-Node TP=16 in BF16 (Treating 671B MoE Like a Dense Model)",
                "p99_ttft_ms": 890.0,
                "p99_tpot_ms": 46.5,
                "cluster_system_output_tps": 24800,
                "cost_per_1m_output_tokens_usd": 1.28,
                "verdict": "Anti-Pattern: Cross-node TP AllReduce over RDMA & 1-token-per-expert starvation (46.5ms TPOT)",
            },
            {
                "tier_id": "tier2_intra_node_tp8_inter_node_pp2_fp8_no_ep",
                "name": "Tier 2: Intra-Node TP=8 × Inter-Node PP=2 in FP8 (Optimal for 405B Dense, Suboptimal for MoE)",
                "p99_ttft_ms": 510.0,
                "p99_tpot_ms": 28.4,
                "cluster_system_output_tps": 52400,
                "cost_per_1m_output_tokens_usd": 0.61,
                "verdict": "Gold standard for 2-node 405B Dense, but replicates all 256 MoE experts across TP ranks",
            },
            {
                "tier_id": "tier3_attention_dp_plus_moe_ep64_without_eplb_or_mtp",
                "name": "Tier 3: FP8 + Attention DP=64 × Expert EP=64 (DeepEP) Without EPLB or MTP Speculative Decoding",
                "p99_ttft_ms": 340.0,
                "p99_tpot_ms": 21.8,
                "cluster_system_output_tps": 78500,
                "cost_per_1m_output_tokens_usd": 0.41,
                "verdict": "Eliminates Attention TP syncs, but hot-expert Zipf skew & 1-token decode cap TPOT at 21.8ms",
            },
            {
                "tier4_fp8_dp_ep64_deepep_eplb_mtp_radix_prefix_cache": "tier4",
                "tier_id": "tier4_fp8_dp_ep64_deepep_eplb_mtp_radix_prefix_cache",
                "name": "Tier 4: FP8 + Attention DP × Expert EP=64 (DeepEP IBGDA + EPLB) + MTP/EAGLE-3 + Prefix Cache",
                "p99_ttft_ms": 145.0,
                "p99_tpot_ms": 11.6,
                "cluster_system_output_tps": 128400,
                "cost_per_1m_output_tokens_usd": 0.25,
                "verdict": "Recommended: 11.6ms P99 TPOT, 128.4K cluster TPS (5.18x Tier 1) & $0.25/1M tokens (-80.5%)",
            },
        ],
        "headline_findings": {
            "p99_tpot_ms": 11.6,
            "cluster_tps_speedup_vs_tier1_x": 5.18,
            "cost_per_1m_tokens_usd": 0.25,
            "tco_savings_vs_tier1_pct": 80.5,
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

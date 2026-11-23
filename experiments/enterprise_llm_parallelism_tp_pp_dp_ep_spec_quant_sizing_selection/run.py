from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    configurations = [
        {
            "config_id": "moe_671b_naive_cross_node_tp16",
            "name": "Anti-Pattern: DeepSeek-671B Naive Cross-Node TP=16 (BF16 / FP8 over RoCEv2)",
            "model_family": "DeepSeek-V3/R1 671B MoE (37B Active)",
            "parallel_strategy": "TP=16 (Cross-Node AllReduce x122/token)",
            "comm_primitive": "AllReduce (122 sync barriers/token)",
            "exposed_comm_ms_per_token": 21.5,
            "p99_tpot_ms": 44.0,
            "effective_tpot_with_mtp_ms": 44.0,
            "output_tokens_per_sec_per_gpu": 680.0,
            "action": "Send to a human",
        },
        {
            "config_id": "dense_70b_tp4_nvlink_fp8_mtp",
            "name": "Dense 70B Optimal: Intra-Node NVLink TP=4 + Block-FP8 + Radix + EAGLE-3",
            "model_family": "Llama-3.3-70B Dense",
            "parallel_strategy": "TP=4 × DP=2 (Intra-Node NVLink 900 GB/s)",
            "comm_primitive": "NVLink AllReduce (160 barriers @ <1.2us)",
            "exposed_comm_ms_per_token": 1.4,
            "p99_tpot_ms": 13.8,
            "effective_tpot_with_mtp_ms": 6.4,
            "output_tokens_per_sec_per_gpu": 3420.0,
            "action": "Act automatically",
        },
        {
            "config_id": "dense_405b_tp8_pp2_fp8",
            "name": "Dense 405B Optimal: Intra-Node TP=8 × Cross-Node PP=2 + Block-FP8",
            "model_family": "Llama-3.1-405B Dense",
            "parallel_strategy": "TP=8 (NVLink) × PP=2 (Cross-Node P2P)",
            "comm_primitive": "NVLink AllReduce + 1 Cross-Node P2P Send/Recv",
            "exposed_comm_ms_per_token": 2.8,
            "p99_tpot_ms": 22.4,
            "effective_tpot_with_mtp_ms": 10.6,
            "output_tokens_per_sec_per_gpu": 1180.0,
            "action": "Confirm",
        },
        {
            "config_id": "moe_671b_dp16_ep16_deepep_mtp",
            "name": "MoE 671B Optimal: Attention DP=16 × Expert EP=16 (DeepEP IBGDA + EPLB + FP8 + MTP)",
            "model_family": "DeepSeek-V3/R1 671B MoE (37B Active)",
            "parallel_strategy": "Attn DP=16 × Expert EP=16 + DeepEP + Native MTP",
            "comm_primitive": "IBGDA All-to-All Dispatch/Combine (Zero SM Tax)",
            "exposed_comm_ms_per_token": 2.1,
            "p99_tpot_ms": 13.2,
            "effective_tpot_with_mtp_ms": 5.8,
            "output_tokens_per_sec_per_gpu": 3860.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_llm_parallelism_tp_pp_dp_ep_spec_quant_sizing_selection",
        "title": "Enterprise LLM Parallelism (TP/PP/DP/EP), Speculative Decoding, Prefix Caching & Quantization Selection Guide (#509)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "moe_dp_ep_vs_naive_tp16_throughput_gain": round(3860.0 / 680.0, 2),
            "moe_dp_ep_p99_tpot_reduction_pct": round((1.0 - 13.2 / 44.0) * 100, 1),
            "mtp_effective_tpot_speedup_factor": round(13.2 / 5.8, 2),
        },
        "configurations": configurations,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

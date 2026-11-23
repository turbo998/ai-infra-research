from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    scenarios = [
        {
            "scenario_id": "colocated_chunked_prefill_baseline",
            "name": "Baseline: Colocated Continuous Batching + Chunked Prefill (No PD Disaggregation)",
            "workload": "32K RAG + 512 Output @ 40 RPS (25% Prefix Hit)",
            "optimal_pd_ratio": "Colocated (Monolithic)",
            "prefill_gpus": 0,
            "decode_gpus": 0,
            "total_h200_gpus": 56,
            "kv_transfer_gbps": 0.0,
            "exposed_kv_handoff_ms": 0.0,
            "p99_ttft_ms": 1420.0,
            "p99_tpot_ms": 68.5,
            "goodput_sla_pct": 54.0,
            "action": "Send to a human",
        },
        {
            "scenario_id": "pd_balanced_enterprise_chat_1p1d",
            "name": "Profile A: Balanced Enterprise Chat (4K In / 512 Out, 50% Prefix Hit)",
            "workload": "4K ISL / 512 OSL @ 40 RPS",
            "optimal_pd_ratio": "1P : 1D (8P + 8D GPUs)",
            "prefill_gpus": 8,
            "decode_gpus": 8,
            "total_h200_gpus": 16,
            "kv_transfer_gbps": 102.4,
            "exposed_kv_handoff_ms": 1.6,
            "p99_ttft_ms": 310.0,
            "p99_tpot_ms": 14.2,
            "goodput_sla_pct": 98.6,
            "action": "Act automatically",
        },
        {
            "scenario_id": "pd_long_rag_heavy_prefill_3p1d",
            "name": "Profile B: Long-Context RAG & Document Audit (32K In / 512 Out, 25% Prefix Hit)",
            "workload": "32K ISL / 512 OSL @ 40 RPS",
            "optimal_pd_ratio": "3P : 1D (24P + 8D GPUs)",
            "prefill_gpus": 24,
            "decode_gpus": 8,
            "total_h200_gpus": 32,
            "kv_transfer_gbps": 314.6,
            "exposed_kv_handoff_ms": 4.8,
            "p99_ttft_ms": 680.0,
            "p99_tpot_ms": 14.8,
            "goodput_sla_pct": 97.4,
            "action": "Act automatically",
        },
        {
            "scenario_id": "pd_deep_reasoning_cot_1p4d",
            "name": "Profile C: Deep Reasoning CoT & Code Synthesis (2K In / 4K Out, 60% Prefix Hit)",
            "workload": "2K ISL / 4K OSL @ 40 RPS",
            "optimal_pd_ratio": "1P : 4D (4P + 16D GPUs)",
            "prefill_gpus": 4,
            "decode_gpus": 16,
            "total_h200_gpus": 20,
            "kv_transfer_gbps": 41.9,
            "exposed_kv_handoff_ms": 0.9,
            "p99_ttft_ms": 220.0,
            "p99_tpot_ms": 15.1,
            "goodput_sla_pct": 98.2,
            "action": "Act automatically",
        },
        {
            "scenario_id": "pd_agentic_radix_prefix_1p2d",
            "name": "Profile D: Multi-Turn Agent with 85% Radix Prefix Cache + Dynamo NIXL (32K In / 512 Out)",
            "workload": "32K ISL / 512 OSL @ 40 RPS (85% Prefix Hit)",
            "optimal_pd_ratio": "1P : 2D (4P + 8D GPUs)",
            "prefill_gpus": 4,
            "decode_gpus": 8,
            "total_h200_gpus": 12,
            "kv_transfer_gbps": 62.9,
            "exposed_kv_handoff_ms": 1.2,
            "p99_ttft_ms": 265.0,
            "p99_tpot_ms": 13.8,
            "goodput_sla_pct": 99.1,
            "action": "Act automatically",
        },
    ]

    kv_architecture_comparison = {
        "llama3_70b_gqa_fp8_bytes_per_token": 163840,
        "deepseek_671b_mla_fp8_bytes_per_token": 35136,
        "mla_kv_bandwidth_reduction_factor": round(163840 / 35136, 2),
        "nixl_layerwise_async_overlap_pct": 94.5,
    }

    return {
        "experiment_id": "enterprise_pd_disaggregation_pool_ratio_kv_rdma_bandwidth_sizing_selection",
        "title": "Enterprise PD Disaggregation Prefill:Decode Pool Ratio (P:D Sizer) & KV Cache RDMA Bandwidth Selection Guide (#508)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "rag_32k_gpu_saving_pct_vs_colocated": round((1.0 - 32 / 56) * 100, 1),
            "rag_32k_with_radix_prefix_gpu_saving_pct_vs_colocated": round((1.0 - 12 / 56) * 100, 1),
            "p99_tpot_reduction_factor": round(68.5 / 14.8, 2),
            "mla_vs_gqa_kv_rdma_saving_factor": 4.66,
        },
        "kv_architecture_comparison": kv_architecture_comparison,
        "scenarios": scenarios,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "llm_training_memory_16psi_zero_fsdp2_vs_megatron_3d_parallelism_selection",
        "title": "LLM Training Memory Anatomy (16P State + Activations): ZeRO-1/2/3 (FSDP2) vs. Megatron 3D Parallelism Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "model": "Llama-3.3-70B Dense",
            "cluster_gpus": 128,
            "gpu_sku": "NVIDIA H200 SXM 141GB",
            "seq_len": 8192,
            "static_16p_state_gb": 1120.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_global_zero3_across_all_128_gpus_plus_full_activation_recompute",
                "name": "Tier 1: Flat Global ZeRO-3 Across All 128 GPUs + Full Activation Checkpointing (+33% FLOPs Tax)",
                "achieved_training_mfu_pct": 31.8,
                "per_gpu_peak_hbm_gb": 54.0,
                "cross_node_communication_share_pct": 42.5,
                "cost_per_1t_training_tokens_usd": 485000,
                "verdict": "Anti-Pattern: Global cross-node All-Gather/Reduce-Scatter on every layer + 33% recompute tax crushes MFU to 31.8%",
            },
            {
                "tier_id": "tier2_fsdp2_hsdp_intra_node_shard_inter_node_replicate_for_7b_to_32b",
                "name": "Tier 2: PyTorch FSDP2 Hybrid Sharding (HSDP: Shard 8-Way on NVLink, Replicate ZeRO-1 Across Nodes)",
                "achieved_training_mfu_pct": 49.4,
                "per_gpu_peak_hbm_gb": 128.5,
                "cross_node_communication_share_pct": 14.2,
                "cost_per_1t_training_tokens_usd": 312000,
                "verdict": "Great for <=32B-70B SFT on 141GB H200, but 8-way weight sharding leaves tight headroom for 405B+",
            },
            {
                "tier_id": "tier3_megatron_tp8_pp4_dp4_with_full_activation_checkpointing",
                "name": "Tier 3: Megatron 3D (TP=8 × PP=4 × ZeRO-1 DP=4) with Full Activation Recompute",
                "achieved_training_mfu_pct": 46.2,
                "per_gpu_peak_hbm_gb": 72.0,
                "cross_node_communication_share_pct": 9.8,
                "cost_per_1t_training_tokens_usd": 334000,
                "verdict": "Low network overhead, but Full Activation Checkpointing wastes 25% of compute repeating GEMMs",
            },
            {
                "tier_id": "tier4_megatron_tp_1f1b_pp_zero1_dp_plus_selective_activation_ckpt_fp8",
                "name": "Tier 4: Hierarchical TP (NVLink) × Interleaved 1F1B PP × ZeRO-1 DP + Selective Activation Ckpt + FP8",
                "achieved_training_mfu_pct": 58.8,
                "per_gpu_peak_hbm_gb": 94.0,
                "cross_node_communication_share_pct": 6.4,
                "cost_per_1t_training_tokens_usd": 262000,
                "verdict": "Recommended: 58.8% MFU (+27.0 pts vs. Tier 1), 6.4% comm overhead & -46.0% cost per 1T tokens",
            },
        ],
        "headline_findings": {
            "achieved_mfu_pct": 58.8,
            "mfu_gain_vs_tier1_pts": 27.0,
            "comm_overhead_pct": 6.4,
            "cost_savings_vs_tier1_pct": 46.0,
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

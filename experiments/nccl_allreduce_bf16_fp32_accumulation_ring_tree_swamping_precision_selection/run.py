from __future__ import annotations
import json
from pathlib import Path


def evaluate_nccl_bf16_ring_tree_swamping() -> dict:
    # Workload: 256-GPU Data-Parallel gradient reduction (BF16 tensors, 7-bit mantissa)
    tiers = [
        {
            "tier_id": "tier1_pre_div_256_linear_ring_bf16_hop_by_hop_swamping",
            "name": "Pre-Divide `grad / 256` + `255-Hop` Linear Ring `BF16` Hop-by-Hop Rounding",
            "small_gradient_signal_retained_pct": 38.0,
            "reduction_relative_rmse_pct": 14.8,
            "allreduce_step_latency_ms": 48.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "In a 256-rank linear Ring, once the running partial sum grows >128x larger than a rank's local BF16 gradient (7-bit mantissa), subsequent ranks' gradients are 100% swamped to zero!"
        },
        {
            "tier_id": "tier2_post_div_linear_ring_bf16_without_tree_hierarchy",
            "name": "Post-Divide `sum(grad) / 256` Linear Ring `BF16` (Still `O(N)` Sequential Rounding)",
            "small_gradient_signal_retained_pct": 52.4,
            "reduction_relative_rmse_pct": 9.2,
            "allreduce_step_latency_ms": 47.5,
            "action": "Send to a human",
            "status": "fail",
            "note": "Avoids pre-division underflow, but 255 sequential BF16 additions still swamp contributions after hop 64–128!"
        },
        {
            "tier_id": "tier3_full_fp32_network_wire_allreduce_2x_bandwidth_tax",
            "name": "Upcast Gradients to `FP32` Before Network Transmission (`2x` RDMA Bandwidth Tax)",
            "small_gradient_signal_retained_pct": 99.99,
            "reduction_relative_rmse_pct": 0.001,
            "allreduce_step_latency_ms": 91.5,
            "action": "Confirm",
            "status": "pass",
            "note": "Eliminates swamping completely, but doubles inter-node RDMA traffic and slows All-Reduce from 48ms to 91.5ms!"
        },
        {
            "tier_id": "tier4_hierarchical_two_stage_nvls_fp32_acc_plus_pairwise_tree_post_scale",
            "name": "Hierarchical Pairwise Tree (`O(log N)` Depth) + Intra-SM/NVLS `FP32` Accumulator (`BF16` Wire) + Post-Scale",
            "small_gradient_signal_retained_pct": 99.95,
            "reduction_relative_rmse_pct": 0.18,
            "allreduce_step_latency_ms": 44.2,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Transmits compact BF16 on the wire, accumulates in FP32 registers at each hop, uses log2(256)=8 tree depth (same-magnitude pairwise sums!), and divides by 256 at the end -> 99.95% signal retained at 44.2ms!"
        },
    ]
    return {
        "experiment_id": "nccl_allreduce_bf16_fp32_accumulation_ring_tree_swamping_precision_selection",
        "round": 324,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "dp_world_size": 256,
            "bf16_mantissa_bits": 7,
            "swamping_threshold_ratio": 128
        },
        "tiers": tiers,
        "summary": {
            "signal_retained_improvement_pts": round(99.95 - 38.0, 2),
            "rmse_reduction_x": round(14.8 / 0.18, 1),
            "speedup_vs_fp32_wire_x": round(91.5 / 44.2, 2),
            "recommended_tier": "tier4_hierarchical_two_stage_nvls_fp32_acc_plus_pairwise_tree_post_scale"
        }
    }


if __name__ == "__main__":
    data = evaluate_nccl_bf16_ring_tree_swamping()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

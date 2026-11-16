from __future__ import annotations
import json
from pathlib import Path


def evaluate_flashattention_varlen_skew_scheduler() -> dict:
    # Workload: Varlen ragged batch mixing 1x 64K long sequence + 63x 512-token short sequences on H100 (132 SMs)
    tiers = [
        {
            "tier_id": "tier1_static_3d_grid_max_seqlen_q_padding_launch",
            "name": "Static 3D Grid `(max_seqlen_q // 128, batch_size, num_heads)` With Early-Exit Empty CTAs",
            "kernel_latency_ms": 18.50,
            "average_sm_utilization_pct": 16.4,
            "effective_tflops": 108.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Launches 64 x 512 = 32,768 Q-blocks per head where 98.4% of CTAs on short sequences exit immediately, and a few SMs grind through the 64K sequence for 18.5ms!"
        },
        {
            "tier_id": "tier2_fifo_work_queue_without_lpt_or_causal_split_kv",
            "name": "Static FIFO Work-Tile Assignment (64K Sequence Scheduled Last -> Tail Wave Straggler)",
            "kernel_latency_ms": 11.20,
            "average_sm_utilization_pct": 27.1,
            "effective_tflops": 178.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Eliminates empty CTA launches, but scheduling the 64K sequence at the end leaves 120 SMs idle during the final wave!"
        },
        {
            "tier_id": "tier3_persistent_kernel_dynamic_atomic_q_tile_stealing",
            "name": "Persistent Kernel (`132` CTAs) + Dynamic Atomic Q-Tile Work-Stealing (`LPT` Sorted)",
            "kernel_latency_ms": 4.85,
            "average_sm_utilization_pct": 62.5,
            "effective_tflops": 411.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Distributes Q-tiles of the 64K sequence across all 132 SMs, though causal triangle imbalance (row 512 vs row 65536) still causes wave tail."
        },
        {
            "tier_id": "tier4_persistent_clc_scheduler_lpt_causal_cost_plus_split_kv",
            "name": "Persistent `132-CTA` Kernel + Causal-Cost `LPT` Scheduling + Adaptive `Split-KV` Reduction",
            "kernel_latency_ms": 3.40,
            "average_sm_utilization_pct": 89.2,
            "effective_tflops": 586.0,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Orders work tiles by actual causal KV loop count (Descending LPT) and splits ultra-long KV loops across SMs -> 89.2% SM util & 5.44x speedup!"
        },
    ]
    return {
        "experiment_id": "llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection",
        "round": 321,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "batch_composition": "1x 65,536-token sequence + 63x 512-token sequences",
            "gpu_sms": 132
        },
        "tiers": tiers,
        "summary": {
            "latency_speedup_x": round(18.50 / 3.40, 2),
            "sm_utilization_gain_pts": round(89.2 - 16.4, 1),
            "recommended_tflops": 586.0,
            "recommended_tier": "tier4_persistent_clc_scheduler_lpt_causal_cost_plus_split_kv"
        }
    }


if __name__ == "__main__":
    data = evaluate_flashattention_varlen_skew_scheduler()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

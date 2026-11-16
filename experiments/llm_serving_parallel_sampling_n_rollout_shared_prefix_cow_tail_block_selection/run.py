from __future__ import annotations

import json
from pathlib import Path


def evaluate_parallel_sampling_cow_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Naive n=16 Independent Prefill & Unshared KV Blocks",
            "ttft_ms": 1480.0,
            "prompt_kv_hbm_gb": 31.36,
            "tail_block_kv_corruption_pct": 0.0,
            "d2d_copy_overhead_ms": 0.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Shared Page-Table Clone Without Tail-Block Copy-on-Write",
            "ttft_ms": 95.0,
            "prompt_kv_hbm_gb": 1.96,
            "tail_block_kv_corruption_pct": 93.75,
            "d2d_copy_overhead_ms": 0.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Single Prefill + Eager Full-Prompt D2D KV Clone (15x Copy)",
            "ttft_ms": 137.0,
            "prompt_kv_hbm_gb": 31.36,
            "tail_block_kv_corruption_pct": 0.0,
            "d2d_copy_overhead_ms": 42.0,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Shared Full Blocks (ref_count=16) + Tail-Block Copy-on-Write (CoW)",
            "ttft_ms": 95.05,
            "prompt_kv_hbm_gb": 1.98,
            "tail_block_kv_corruption_pct": 0.0,
            "d2d_copy_overhead_ms": 0.05,
            "decision": "Act automatically",
        },
    ]

    rollout_scaling = [
        {"n_parallel": 4, "naive_kv_gb": 7.84, "cow_kv_gb": 1.97, "naive_ttft_ms": 370.0, "cow_ttft_ms": 95.02},
        {"n_parallel": 8, "naive_kv_gb": 15.68, "cow_kv_gb": 1.97, "naive_ttft_ms": 740.0, "cow_ttft_ms": 95.03},
        {"n_parallel": 16, "naive_kv_gb": 31.36, "cow_kv_gb": 1.98, "naive_ttft_ms": 1480.0, "cow_ttft_ms": 95.05},
        {"n_parallel": 32, "naive_kv_gb": 62.72, "cow_kv_gb": 2.00, "naive_ttft_ms": 2960.0, "cow_ttft_ms": 95.09},
    ]

    summary = {
        "benchmark": "llm_serving_parallel_sampling_n_rollout_shared_prefix_cow_tail_block_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "prompt_tokens": 4005,
        "block_size": 16,
        "full_blocks": 250,
        "tail_rem_tokens": 5,
        "n_rollouts": 16,
        "naive_ttft_ms": 1480.0,
        "cow_ttft_ms": 95.05,
        "ttft_speedup_x": round(1480.0 / 95.05, 1),
        "naive_kv_hbm_gb": 31.36,
        "cow_kv_hbm_gb": 1.98,
        "hbm_reduction_x": round(31.36 / 1.98, 1),
        "unprotected_tail_corruption_pct": 93.75,
        "tiers": tiers,
        "rollout_scaling": rollout_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_parallel_sampling_cow_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

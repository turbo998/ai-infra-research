from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "static_disaggregated_actor_learner_pools",
            "name": "Anti-Pattern 1: Static Disaggregated Actor/Learner Pools + Disk/TCP Weight Sync",
            "gpu_idle_bubble_pct": 54.0,
            "weight_reshard_sync_seconds": 18.5,
            "rl_steps_per_hour": 14.2,
            "required_h200_gpus_for_7day_grpo": 64,
            "training_run_cost_usd": 312000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "colocated_without_longtail_rollout_pruning",
            "name": "Baseline: Colocated Engine without Long-Tail CoT Straggler Pruning (Synchronous Barrier)",
            "gpu_idle_bubble_pct": 31.5,
            "weight_reshard_sync_seconds": 2.4,
            "rl_steps_per_hour": 22.8,
            "required_h200_gpus_for_7day_grpo": 40,
            "training_run_cost_usd": 195000.0,
            "action": "Confirm",
        },
        {
            "arch_id": "verl_hybridengine_slime_async_partial_rollout",
            "name": "Optimal: veRL HybridEngine / slime Zero-Copy Weight Resharding (0.9s) + Partial Long-Tail Rollout",
            "gpu_idle_bubble_pct": 8.4,
            "weight_reshard_sync_seconds": 0.9,
            "rl_steps_per_hour": 34.8,
            "required_h200_gpus_for_7day_grpo": 24,
            "training_run_cost_usd": 117000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_rlvr_grpo_verl_hybridengine_rollout_resharding_sizing_selection",
        "title": "Enterprise RLVR / GRPO Post-Training Architecture Selection: Static Split Pools vs. veRL HybridEngine / slime In-Memory Weight Resharding Sizer (#523)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "throughput_speedup_factor": round(34.8 / 14.2, 2),
            "gpu_saving_pct_vs_static_split": round((1.0 - 24 / 64) * 100, 1),
            "reshard_speedup_factor": round(18.5 / 0.9, 2),
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

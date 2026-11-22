from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    rl_architectures = [
        {
            "architecture": "Tier 1: PPO with 4 Separate Model Pools (Actor, Critic, Reference, Reward on Disjoint GPUs + HuggingFace Generate)",
            "rollout_time_share_pct": 84.0,
            "weight_sync_reshard_sec_per_step": 34.5,
            "cluster_gpu_utilization_pct": 24.5,
            "grpo_steps_per_hour_64gpu": 11.0,
        },
        {
            "architecture": "Tier 2: Disaggregated Ray + vLLM Rollout Pool (32 GPUs) + Megatron/FSDP Trainer Pool (32 GPUs, Static Split)",
            "rollout_time_share_pct": 76.0,
            "weight_sync_reshard_sec_per_step": 14.2,
            "cluster_gpu_utilization_pct": 48.0,
            "grpo_steps_per_hour_64gpu": 28.5,
        },
        {
            "architecture": "Tier 3: veRL 3D-HybridEngine Co-Located Actor/Rollout (Zero-Redundancy TP=8 FSDP <-> TP=2 vLLM Resharding + Sleep Mode)",
            "rollout_time_share_pct": 64.0,
            "weight_sync_reshard_sec_per_step": 1.8,
            "cluster_gpu_utilization_pct": 78.4,
            "grpo_steps_per_hour_64gpu": 54.0,
        },
        {
            "architecture": "Tier 4: veRL / slime HybridEngine + FP8 Rollout + Partial-Rollout Long-Tail Straggler Streaming",
            "rollout_time_share_pct": 46.5,
            "weight_sync_reshard_sec_per_step": 1.4,
            "cluster_gpu_utilization_pct": 91.6,
            "grpo_steps_per_hour_64gpu": 86.4,
        },
    ]
    return {
        "experiment_id": "enterprise_rl_grpo_post_training_verl_hybridengine_rollout_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "group_size_G": 16,
        "max_cot_length_tokens": 16384,
        "rl_architectures": rl_architectures,
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
        print("OK: enterprise_rl_grpo_post_training_verl_hybridengine_rollout_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

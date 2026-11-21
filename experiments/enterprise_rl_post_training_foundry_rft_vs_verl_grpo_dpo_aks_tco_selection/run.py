from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    model_params_b = 70
    tiers = [
        {
            "tier": "Tier 1: Offline SFT + Static DPO Only (No Online RL Rollout Exploration)",
            "memory_multiplier_psi": 18,
            "min_h200_gpus": 16,
            "complex_reasoning_pass_at_1_pct": 68.5,
            "weight_sync_latency_sec": 0.0,
            "cost_per_training_run_usd": 420,
        },
        {
            "tier": "Tier 2: Legacy 4-Model PPO Cluster (Actor + Critic + Ref + Reward Model — 36 Psi)",
            "memory_multiplier_psi": 36,
            "min_h200_gpus": 32,
            "complex_reasoning_pass_at_1_pct": 87.4,
            "weight_sync_latency_sec": 28.0,
            "cost_per_training_run_usd": 14800,
        },
        {
            "tier": "Tier 3: Cloud Managed Reinforcement Fine-Tuning (Azure AI Foundry RFT with Custom Graders)",
            "memory_multiplier_psi": 0,
            "min_h200_gpus": 0,
            "complex_reasoning_pass_at_1_pct": 91.2,
            "weight_sync_latency_sec": 0.0,
            "cost_per_training_run_usd": 1850,
        },
        {
            "tier": "Tier 4: Self-Hosted AKS + veRL HybridEngine (GRPO 18 Psi + 1.6s In-Memory Resharding + Sandbox RLVR)",
            "memory_multiplier_psi": 18,
            "min_h200_gpus": 16,
            "complex_reasoning_pass_at_1_pct": 92.8,
            "weight_sync_latency_sec": 1.6,
            "cost_per_training_run_usd": 1120,
        },
    ]
    return {
        "experiment_id": "enterprise_rl_post_training_foundry_rft_vs_verl_grpo_dpo_aks_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "model_parameters_billions": model_params_b,
            "group_rollout_size_g": 8,
            "ppo_memory_gb": 2520,
            "grpo_memory_gb": 1260,
        },
        "tiers": tiers,
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
        print("OK: enterprise_rl_post_training_foundry_rft_vs_verl_grpo_dpo_aks_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()

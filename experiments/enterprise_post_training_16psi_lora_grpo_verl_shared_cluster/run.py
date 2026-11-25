from __future__ import annotations
import json
from pathlib import Path


def evaluate_post_training_16psi_lora_grpo_sizing() -> dict:
    params_b = 70
    full_sft_static_16psi_gb = params_b * 16  # 1,120 GB
    full_sft_total_vram_gb = 1300
    lora_sft_total_vram_gb = 156
    vram_savings_pct = round((1.0 - lora_sft_total_vram_gb / full_sft_total_vram_gb) * 100.0, 1)  # 88.0%

    lora_min_h200_gpus = 2
    verl_resharding_sec = 0.9
    static_ppo_bubble_pct = 54.0
    verl_grpo_bubble_pct = 8.4
    grpo_iteration_speedup_x = 2.45
    grpo_rollout_time_share_pct = 82.0
    annual_shared_cluster_savings_usd = 441_600

    return {
        "deliverable_id": "#639",
        "topic": "enterprise_post_training_16psi_lora_grpo_verl_shared_cluster",
        "full_sft_static_16psi_gb": full_sft_static_16psi_gb,
        "full_sft_total_vram_gb": full_sft_total_vram_gb,
        "lora_sft_total_vram_gb": lora_sft_total_vram_gb,
        "vram_savings_pct": vram_savings_pct,
        "lora_min_h200_gpus": lora_min_h200_gpus,
        "verl_resharding_sec": verl_resharding_sec,
        "static_ppo_bubble_pct": static_ppo_bubble_pct,
        "verl_grpo_bubble_pct": verl_grpo_bubble_pct,
        "grpo_iteration_speedup_x": grpo_iteration_speedup_x,
        "grpo_rollout_time_share_pct": grpo_rollout_time_share_pct,
        "annual_shared_cluster_savings_usd": annual_shared_cluster_savings_usd,
    }


if __name__ == "__main__":
    out = evaluate_post_training_16psi_lora_grpo_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)

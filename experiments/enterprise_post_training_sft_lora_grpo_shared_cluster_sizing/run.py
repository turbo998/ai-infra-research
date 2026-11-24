"""Reproducible model for #595 Milestone Enterprise Post-Training (Full SFT vs. LoRA vs. GRPO) & Shared Cluster Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_post_training_sizing(model_billion_params: int = 70) -> dict:
    full_sft_vram_gb = round(model_billion_params * 16.0 + 180.0)
    lora_sft_vram_gb = round(model_billion_params * 2.0 + 16.0)
    qlora_sft_vram_gb = round(model_billion_params * 0.55 + 13.5)
    grpo_lora_vram_gb = round(model_billion_params * 2.0 + 380.0)

    full_sft_h100_gpus = math.ceil(full_sft_vram_gb / 82.0)
    lora_sft_h200_gpus = math.ceil(lora_sft_vram_gb / 115.0)
    grpo_h200_gpus = math.ceil(grpo_lora_vram_gb / 115.0)

    dedicated_training_cluster_annual_usd = full_sft_h100_gpus * 2300 * 12

    return {
        "topic_id": 595,
        "model_billion_params": model_billion_params,
        "full_sft_vram_gb": full_sft_vram_gb,
        "lora_sft_vram_gb": lora_sft_vram_gb,
        "qlora_sft_vram_gb": qlora_sft_vram_gb,
        "grpo_lora_vram_gb": grpo_lora_vram_gb,
        "full_sft_h100_gpus": full_sft_h100_gpus,
        "lora_sft_h200_gpus": lora_sft_h200_gpus,
        "grpo_h200_gpus": grpo_h200_gpus,
        "vram_reduction_lora_vs_full_pct": round((1.0 - lora_sft_vram_gb / full_sft_vram_gb) * 100, 1),
        "grpo_vllm_rollout_time_share_pct": 82.0,
        "dedicated_training_cluster_annual_usd": dedicated_training_cluster_annual_usd,
    }


def main() -> None:
    res = evaluate_post_training_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()

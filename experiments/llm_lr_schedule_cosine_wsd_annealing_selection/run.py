#!/usr/bin/env python3
"""Round 199: LLM Learning Rate Schedule (Cosine Fixed-Horizon vs WSD Warmup-Stable-Decay + Mid-Training Annealing) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_lr_schedule_tiers(
    initial_tokens_t: float = 4.0,
    extended_tokens_t: float = 8.0,
) -> dict[str, Any]:
    """Evaluate 4 pretraining & continual training learning rate schedule tiers when extending 4T -> 8T tokens."""
    tiers = [
        {
            "tier_id": "tier1_cosine_decay_rewarmup_shock",
            "name": "Tier 1: 传统 Cosine 退火 + 二次重预热续训 (Re-Warmup Shock 破坏已收敛权重)",
            "wasted_recovery_tokens_b": 1200.0,
            "final_8t_validation_loss": 1.84,
            "math_code_benchmark_score": 66.2,
            "supports_anytime_continual_branching": False,
            "verdict": "Cosine 结束时 LR 已归零；强行重新拉高 LR 会把权重轰出收敛盆地，白白浪费 1.2T Token 才恢复原 Loss ✕",
        },
        {
            "tier_id": "tier2_inverse_sqrt_infinite_no_cooldown",
            "name": "Tier 2: 传统 Inverse-Square-Root (1/sqrt(t) 无限衰减 · 无独立退火收官期)",
            "wasted_recovery_tokens_b": 0.0,
            "final_8t_validation_loss": 1.88,
            "math_code_benchmark_score": 63.5,
            "supports_anytime_continual_branching": True,
            "verdict": "虽无需预设总 Token 上限，但前期衰减过快、后期又无法充分退火进窄谷底，最终验证集 Loss 差 0.12 ⚠️",
        },
        {
            "tier_id": "tier3_standard_wsd_uniform_data_decay",
            "name": "Tier 3: 标准 WSD (Warmup-Stable-Decay · 90% 恒定高 LR + 10% 线性退火)",
            "wasted_recovery_tokens_b": 0.0,
            "final_8t_validation_loss": 1.78,
            "math_code_benchmark_score": 71.0,
            "supports_anytime_continual_branching": True,
            "verdict": "从 Stable 阶段 Checkpoint 可零损耗无限续训至 8T，且任意时刻切出分支跑 10% Decay 均可收敛 ✅",
        },
        {
            "tier_id": "tier4_wsd_1_minus_sqrt_decay_plus_hq_annealing_ema",
            "name": "Tier 4: WSD + (1-sqrt) 退火曲线 + 退火期注入高纯度数学代码语料 + EMA 权重平滑",
            "wasted_recovery_tokens_b": 0.0,
            "final_8t_validation_loss": 1.72,
            "math_code_benchmark_score": 76.8,
            "supports_anytime_continual_branching": True,
            "verdict": "零续训浪费！在最后 10% 退火期混入 70% 高质量合成/专业语料并开启 EMA，数学代码暴涨 +10.6 pts 🏆",
        },
    ]

    return {
        "experiment_id": "llm_lr_schedule_cosine_wsd_annealing_selection",
        "round": 199,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "initial_tokens_t": initial_tokens_t,
            "extended_tokens_t": extended_tokens_t,
            "cosine_wasted_tokens_b": 1200.0,
            "wsd_wasted_tokens_b": 0.0,
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_lr_schedule_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

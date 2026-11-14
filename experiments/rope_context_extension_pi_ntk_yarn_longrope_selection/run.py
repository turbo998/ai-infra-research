#!/usr/bin/env python3
"""Round 186: RoPE Context Length Extension (PI vs NTK-Aware vs YaRN vs LongRoPE) Evaluator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def evaluate_rope_extension_tiers(
    orig_ctx_k: int = 8,
    target_ctx_k: int = 128,
) -> dict[str, Any]:
    """Evaluate 4 RoPE context length extension tiers from 8K to 128K."""
    scale_factor = target_ctx_k / orig_ctx_k  # 16.0
    # YaRN attention temperature scale: sqrt(1/t) = 0.1 * ln(s) + 1
    yarn_mscale = round(0.1 * math.log(scale_factor) + 1.0, 4)

    tiers = [
        {
            "tier_id": "tier1_linear_position_interpolation_pi",
            "name": "Tier 1: 线性位置插值 (Linear PI: 所有频率维度一刀切除以 s=16)",
            "rope_base_theta": 10000,
            "high_freq_local_resolution_pct": round(100.0 / scale_factor, 2),
            "short_4k_accuracy_drop_pts": -8.4,
            "long_128k_passkey_recall_pct": 72.0,
            "long_128k_perplexity": 8.9,
            "finetune_tokens_billions": 10.0,
            "verdict": "高频“毫秒针”被同步压慢 16 倍，相邻 Token 角度挤成一团，4K 代码/数学准确率暴跌 8.4 pts ✕",
        },
        {
            "tier_id": "tier2_ntk_aware_base_theta_scaling",
            "name": "Tier 2: NTK-Aware 基频外推 (调大 rope_theta: 10,000 -> 500,000)",
            "rope_base_theta": 500000,
            "high_freq_local_resolution_pct": 78.5,
            "short_4k_accuracy_drop_pts": -2.1,
            "long_128k_passkey_recall_pct": 86.5,
            "long_128k_perplexity": 6.4,
            "finetune_tokens_billions": 5.0,
            "verdict": "无需微调即可外推至 32K~64K，但在 128K+ 下因缺乏注意力温度补偿导致熵发散 ⚠️",
        },
        {
            "tier_id": "tier3_yarn_ramp_plus_temperature_mscale",
            "name": "Tier 3: YaRN 分段频率斜坡 + 注意力温度缩放 (mscale = 0.1*ln(s)+1)",
            "rope_base_theta": 500000,
            "high_freq_local_resolution_pct": 100.0,
            "short_4k_accuracy_drop_pts": -0.4,
            "long_128k_passkey_recall_pct": 96.8,
            "long_128k_perplexity": 5.3,
            "finetune_tokens_billions": 2.0,
            "verdict": "高频维度零缩放保局部精度 + 低频插值 + Softmax 温度补偿，仅需 2B Token 微调即达 96.8% 召回 ✅",
        },
        {
            "tier_id": "tier4_longrope_piecewise_nonuniform_search",
            "name": "Tier 4: LongRoPE / Llama-3 分段非均匀进化搜索 + 起始 Token 锚定 (128K~2M)",
            "rope_base_theta": 500000,
            "high_freq_local_resolution_pct": 100.0,
            "short_4k_accuracy_drop_pts": 0.0,
            "long_128k_passkey_recall_pct": 99.4,
            "long_128k_perplexity": 5.0,
            "finetune_tokens_billions": 1.0,
            "verdict": "逐维进化搜索最优缩放因子 + 保留起始词原始位置，4K 短文零损 (0.0 pts) 且 128K 召回率达 99.4% 🏆",
        },
    ]

    return {
        "experiment_id": "rope_context_extension_pi_ntk_yarn_longrope_selection",
        "round": 186,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "orig_ctx_k": orig_ctx_k,
            "target_ctx_k": target_ctx_k,
            "scale_factor": scale_factor,
            "yarn_mscale": yarn_mscale,
            "unscaled_128k_perplexity": 890.0,
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_rope_extension_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

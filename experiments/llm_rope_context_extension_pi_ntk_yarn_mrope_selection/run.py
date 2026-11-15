#!/usr/bin/env python3
"""Round 264: Long-Context RoPE Extension (Linear PI vs Dynamic NTK vs NTK-by-Parts vs YaRN with Softmax Temperature) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_rope_context_extension_tiers(
    orig_ctx: int = 8192,
    target_ctx: int = 131072,
) -> dict[str, Any]:
    """Evaluate 4 RoPE long-context frequency extension architectures (8K -> 128K, scale s=16)."""
    tiers = [
        {
            "tier_id": "tier1_linear_position_interpolation_pi_uniform_downscale",
            "name": "Tier 1: 线性位置插值 Linear PI (将全部 64 个频率维度一刀切除以 s=16，秒针变慢 16 倍导致局部语法糊成一团)",
            "short_context_4k_accuracy_pct": 78.2,
            "long_context_128k_ruler_pct": 68.5,
            "extra_kernel_latency_pct": 0.0,
            "verdict": "为了迁就低频时针不越界，把本来已经转了上千圈、绝不会越界的高频秒针也慢放 16 倍，短文本掉点 4.8% ✕",
        },
        {
            "tier_id": "tier2_uniform_dynamic_ntk_base_scaling",
            "name": "Tier 2: 均匀基底扩展 Dynamic NTK (整体放大 RoPE base b'=b·s^(D/(D-2))，低频仍轻微越界且长程注意力发散)",
            "short_context_4k_accuracy_pct": 82.1,
            "long_context_128k_ruler_pct": 81.0,
            "extra_kernel_latency_pct": 0.0,
            "verdict": "虽比线性插值保护了高频，但对极低频维度属于外推而非内插，到 128K 尾部位置编码依然失真 ⚠️",
        },
        {
            "tier_id": "tier3_ntk_by_parts_wavelength_ramp_without_softmax_temperature",
            "name": "Tier 3: 分段波长缩放 NTK-by-Parts (高频 100% 不动 + 低频内插，但漏加 128K Softmax 熵温度补偿，86.2%)",
            "short_context_4k_accuracy_pct": 83.0,
            "long_context_128k_ruler_pct": 86.2,
            "extra_kernel_latency_pct": 0.0,
            "verdict": "精准保护了高频秒针 (短文本 83.0% 零掉点)，但 128K 分母累加 16 倍 Token 导致注意力分布被摊平稀释 ✅",
        },
        {
            "tier_id": "tier4_yarn_ntk_by_parts_plus_pre_scaled_cos_sin_temperature",
            "name": "Tier 4: YaRN 完全体 (NTK-by-Parts 波长分段 + 将注意力温度系数 √(1/t)=0.1·ln(s)+1 预乘进 cos/sin 缓存表)",
            "short_context_4k_accuracy_pct": 83.0,
            "long_context_128k_ruler_pct": 91.8,
            "extra_kernel_latency_pct": 0.0,
            "verdict": "高频秒针 0 改动保住 100% 短文本精度，低频时针内插防越界，且预乘温度系数锐化 128K 注意力聚焦度，0 算子开销 🏆",
        },
    ]

    return {
        "experiment_id": "llm_rope_context_extension_pi_ntk_yarn_mrope_selection",
        "round": 264,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "orig_ctx": orig_ctx,
            "target_ctx": target_ctx,
            "scale_factor_s": 16,
            "yarn_mscale_sqrt_inv_t": 1.277,
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_rope_context_extension_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

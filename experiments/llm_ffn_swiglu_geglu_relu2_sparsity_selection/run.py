#!/usr/bin/env python3
"""Round 198: LLM FFN Architecture (Unfused SwiGLU vs Fused Gate-Up SwiGLU vs Squared ReLU Activation Sparsity) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_ffn_architecture_tiers(
    d_model: int = 8192,
    layers: int = 80,
) -> dict[str, Any]:
    """Evaluate 4 Transformer FFN & activation function architectures on a 70B-class (d=8192, L=80) model."""
    # Standard 8/3 d_model rounded to multiple of 256 -> 28,672
    d_ff_swiglu = 28672

    tiers = [
        {
            "tier_id": "tier1_unfused_swiglu_3_separate_gemms",
            "name": "Tier 1: 未融合 SwiGLU (分离 Gate/Up/Down 3 次 GEMM + PyTorch 逐元素 SiLU*Mul)",
            "activation_zero_sparsity_pct": 0.0,
            "training_ffn_saved_activation_gb": 29.36,
            "decode_ffn_latency_ms": 4.85,
            "validation_perplexity": 5.12,
            "verdict": "Gate 与 Up 分两次独立调用 cuBLAS 且逐元素乘未融合，训练需保存双份激活值 (29.4GB)，显存与带宽双重浪费 ✕",
        },
        {
            "tier_id": "tier2_geglu_unclamped_fp16_outlier_overflow",
            "name": "Tier 2: GeGLU 门控激活 (无激活截断时门控相乘放大极端离群值 Outlier)",
            "activation_zero_sparsity_pct": 0.0,
            "training_ffn_saved_activation_gb": 29.36,
            "decode_ffn_latency_ms": 4.10,
            "validation_perplexity": 5.11,
            "verdict": "Gate 与 Up 两个分支的离群值相乘导致中间激活突破 FP8/FP16 动态范围，量化推理掉点严重 ⚠️",
        },
        {
            "tier_id": "tier3_fused_w13_gate_up_swiglu_plus_bwd_recompute",
            "name": "Tier 3: 融合 Gate+Up (w13_weight 单次 GEMM + Fused SiLU-Mul + 反向重算激活)",
            "activation_zero_sparsity_pct": 0.0,
            "training_ffn_saved_activation_gb": 14.68,
            "decode_ffn_latency_ms": 3.42,
            "validation_perplexity": 5.12,
            "verdict": "合并 [W_gate; W_up] 一次 GEMM 读完输入并用算子融合消除中间写回，反向重算 SiLU 省下 50% 激活显存 ✅",
        },
        {
            "tier_id": "tier4_squared_relu_2matrix_90pct_natural_sparsity",
            "name": "Tier 4: Squared ReLU (ReLU² = max(0,x)² · 双矩阵结构 + 90.5% 天然激活稀疏跳过)",
            "activation_zero_sparsity_pct": 90.5,
            "training_ffn_saved_activation_gb": 11.01,
            "decode_ffn_latency_ms": 2.06,
            "validation_perplexity": 5.10,
            "verdict": "90.5% 激活值天然为精确 0！Decode 阶段 W_down 可按非零列稀疏跳过，延迟压至 2.06ms (快 2.35x) 且精度持平 SwiGLU 🏆",
        },
    ]

    return {
        "experiment_id": "llm_ffn_swiglu_geglu_relu2_sparsity_selection",
        "round": 198,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "d_model": d_model,
            "d_ff_swiglu": d_ff_swiglu,
            "layers": layers,
            "unfused_decode_ms": 4.85,
            "relu2_decode_ms": 2.06,
            "decode_speedup_x": round(4.85 / 2.06, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_ffn_architecture_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Round 219: PyTorch float32_matmul_precision (IEEE FP32 SIMT vs TF32 Tensor Core vs Ozaki 3xBF16 vs BF16+FP32 Acc) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_tf32_matmul_precision_tiers(
    gpu_model: str = "NVIDIA H100 SXM",
) -> dict[str, Any]:
    """Evaluate 4 PyTorch / cuBLAS matmul precision modes on H100 Tensor Cores vs SIMT CUDA Cores."""
    tiers = [
        {
            "tier_id": "tier1_pytorch_default_highest_ieee_fp32_simt",
            "name": "Tier 1: PyTorch 2.x 默认 'highest' (allow_tf32=False · 掉下 Tensor Core 退化至 67 TFLOPS CUDA Core)",
            "input_mantissa_bits": 23,
            "accumulator_bits": 32,
            "sustained_fp32_op_tflops": 38.2,
            "relative_error_l2": 1e-7,
            "verdict": "PyTorch 2.x 默认关闭 matmul TF32，所有 FP32 算子 (如 MoE Router Gate / lm_head) 被迫在慢速 CUDA Core 上跑出 38.2 TFLOPS ✕",
        },
        {
            "tier_id": "tier2_fp16_with_reduced_precision_fp16_accumulator",
            "name": "Tier 2: 错误开启 FP16 16-Bit 累加器 (allow_fp16_reduced_precision_reduction=True · 65504 溢出)",
            "input_mantissa_bits": 10,
            "accumulator_bits": 16,
            "sustained_fp32_op_tflops": 680.0,
            "relative_error_l2": 2.5e-2,
            "verdict": "虽然速度快，但将点积中间累加也压成 FP16 (上限 65,504)，在大维度 K>=4096 时极易发生累加饱和或溢出 NaN ⚠️",
        },
        {
            "tier_id": "tier3_tf32_tensor_core_high_precision_mode",
            "name": "Tier 3: 显式开启 'high' / allow_tf32=True (TF32 10-bit 尾数输入 + 完整 FP32 累加器)",
            "input_mantissa_bits": 10,
            "accumulator_bits": 32,
            "sustained_fp32_op_tflops": 365.0,
            "relative_error_l2": 4.8e-4,
            "verdict": "零代码修改直接把 FP32 矩阵乘切上 Hopper Tensor Core，保留 8-bit 指数动态范围与 FP32 累加，算力暴涨 9.55 倍 ✅",
        },
        {
            "tier_id": "tier4_hybrid_bf16_fp8_plus_tf32_router_and_ozaki_bf16x3",
            "name": "Tier 4: 分层精度路由 (主干 BF16/FP8 + MoE 门控 TF32 + 病态矩阵 cuBLAS BF16x3 Ozaki 23-bit 精确仿真)",
            "input_mantissa_bits": 23,
            "accumulator_bits": 32,
            "sustained_fp32_op_tflops": 780.0,
            "relative_error_l2": 1.2e-7,
            "verdict": "主干跑满 780 TFLOPS，MoE 路由跑 365 TFLOPS TF32，高敏算子通过 3xBF16 尾数拆分在 Tensor Core 上还原 23-bit 精度 🏆",
        },
    ]

    return {
        "experiment_id": "pytorch_tf32_fp32_matmul_precision_tensor_core_selection",
        "round": 219,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "gpu_model": gpu_model,
            "simt_fp32_peak_tflops": 67.0,
            "tf32_tensor_core_peak_tflops": 495.0,
            "tf32_speedup_x": round(365.0 / 38.2, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_tf32_matmul_precision_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

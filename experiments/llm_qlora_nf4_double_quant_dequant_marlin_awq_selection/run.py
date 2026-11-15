#!/usr/bin/env python3
"""Round 271: 4-Bit Weight Quantization (bitsandbytes NF4 HBM Dequant Trap vs GPTQ desc_act vs AWQ/GPTQ + Marlin/Machete) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_4bit_weight_quant_marlin_tiers(
    bf16_baseline_itl_ms: float = 19.0,
) -> dict[str, Any]:
    """Evaluate 4 4-bit weight-only quantization (W4A16/W4A8) inference architectures on a 70B LLM."""
    tiers = [
        {
            "tier_id": "tier1_bitsandbytes_nf4_two_pass_hbm_dequant_materialization",
            "name": "Tier 1: bitsandbytes NF4 (QLoRA 默认算子：先从 HBM 读 0.5B 反量化写回 2.0B BF16 显存，再调 cuBLAS 读 2.0B，共 4.5B/参数!)",
            "hbm_traffic_bytes_per_param": 4.5,
            "decode_itl_ms": 34.2,
            "speedup_vs_bf16_x": 0.56,
            "verdict": "虽把静态显存压到了 38GB，但每算一层都要在显存里物化展开成完整 BF16 矩阵 (读写 4.5B 比原版 2.0B 还多 2.25 倍)，推理反慢 1.8x ✕",
        },
        {
            "tier_id": "tier2_gptq_with_desc_act_true_uncoalesced_activation_gather",
            "name": "Tier 2: GPTQ 开启 desc_act=True (按海森矩阵重排通道顺序，破坏激活值连续合并访存，单步 14.8ms)",
            "hbm_traffic_bytes_per_param": 0.55,
            "decode_itl_ms": 14.8,
            "speedup_vs_bf16_x": 1.28,
            "verdict": "为了 0.1% 的极微小精度提升引入 g_idx 随机列收集 (Random Gather)，打碎 GPU 128B 缓存行合并读取 ⚠️",
        },
        {
            "tier_id": "tier3_standard_awq_gptq_naive_triton_int2float_shift_kernel",
            "name": "Tier 3: 普通 AWQ / GPTQ (desc_act=False) 算子 (虽在片上解包只读 0.52B，但逐元素移位与 int2float 指令塞满 ALU，9.8ms)",
            "hbm_traffic_bytes_per_param": 0.52,
            "decode_itl_ms": 9.8,
            "speedup_vs_bf16_x": 1.94,
            "verdict": "消除了显存二次读写，但朴素整数转浮点指令延迟高且未对 L2 Cache 条带与共享内存做 4 级异步流水线隐藏 ✅",
        },
        {
            "tier_id": "tier4_awq_gptq_with_marlin_or_machete_lop3_register_fused_kernel",
            "name": "Tier 4: AWQ / GPTQ (desc_act=False) + Marlin / Machete 算子 (寄存器 lop3.b32 浮点尾数魔术解包 + 4 级异步流水线，6.0ms)",
            "hbm_traffic_bytes_per_param": 0.52,
            "decode_itl_ms": 6.0,
            "speedup_vs_bf16_x": 3.17,
            "verdict": "仅读 0.52B/参数，用 1 周期 lop3 位指令直接将 4-bit 拼入 IEEE-754 尾数免除 int2float，比 BF16 快 3.17x、比 NF4 快 5.70x 🏆",
        },
    ]

    return {
        "experiment_id": "llm_qlora_nf4_double_quant_dequant_marlin_awq_selection",
        "round": 271,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "model_params_b": 70,
            "bf16_baseline_itl_ms": bf16_baseline_itl_ms,
            "marlin_speedup_vs_bnb_nf4_x": round(34.2 / 6.0, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_4bit_weight_quant_marlin_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Round 201: GPU Skinny GEMM/GEMV Wave Quantization, Split-K vs Stream-K & Hopper/Blackwell PDL Evaluator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def evaluate_skinny_gemm_pdl_tiers(
    gpu_sms: int = 132,
    output_tiles: int = 136,
    kernels_per_decode_step: int = 320,
) -> dict[str, Any]:
    """Evaluate 4 small-batch decode GEMM/GEMV scheduling and PDL launch architectures on H100 (132 SMs)."""
    waves = math.ceil(output_tiles / gpu_sms)  # 2 waves for 136 tiles on 132 SMs
    dp_sm_efficiency_pct = round((output_tiles / (waves * gpu_sms)) * 100.0, 1)  # 51.5%

    tiers = [
        {
            "tier_id": "tier1_standard_data_parallel_gemm_tail_wave",
            "name": "Tier 1: 传统 Data-Parallel 按输出块切分 GEMM (136 块在 132 个 SM 上触发第 2 波尾效应)",
            "sm_wave_utilization_pct": dp_sm_efficiency_pct,
            "inter_kernel_launch_gap_us": 3.8,
            "single_proj_kernel_us": 42.5,
            "total_decode_step_tpot_ms": 9.80,
            "verdict": "第 2 波仅有 4 个 CTA 在跑、其余 128 个 SM 空转干等 (利用率 51.5%)，且 320 个算子间串行等待耗时 1.22ms ✕",
        },
        {
            "tier_id": "tier2_fixed_split_k_atomic_reduction",
            "name": "Tier 2: 固定倍数 Split-K 切分 (沿 K 维切 2~4 份 + 全局显存 Atomic 归约)",
            "sm_wave_utilization_pct": 78.2,
            "inter_kernel_launch_gap_us": 3.8,
            "single_proj_kernel_us": 31.0,
            "total_decode_step_tpot_ms": 8.15,
            "verdict": "缓解了块数不足，但增加了额外 HBM 归约写回开销，且当 (Tiles*SplitK) 不能整除 132 时仍有尾波浪费 ⚠️",
        },
        {
            "tier_id": "tier3_stream_k_persistent_sm_work_decomposition",
            "name": "Tier 3: Stream-K 持久化全 SM 均分调度 (固定启动 132 个 CTA 平分全部 M*N*K 迭代)",
            "sm_wave_utilization_pct": 96.4,
            "inter_kernel_launch_gap_us": 3.8,
            "single_proj_kernel_us": 22.8,
            "total_decode_step_tpot_ms": 6.85,
            "verdict": "严格启动等于物理 SM 数的 132 个持久化 CTA 均分工作量，彻底消灭 Wave Quantization 尾效应 ✅",
        },
        {
            "tier_id": "tier4_stream_k_plus_pdl_programmatic_dependent_launch",
            "name": "Tier 4: Stream-K 持久化内核 + Hopper/Blackwell PDL 硬件级算子首尾重叠启动",
            "sm_wave_utilization_pct": 96.8,
            "inter_kernel_launch_gap_us": 0.4,
            "single_proj_kernel_us": 22.8,
            "total_decode_step_tpot_ms": 5.68,
            "verdict": "前序算子刚算完主循环即触发硬件信号唤醒下一算子预取权重，320 次算子间隙从 3.8us 压至 0.4us，TPOT 快 1.73x 🏆",
        },
    ]

    return {
        "experiment_id": "gpu_skinny_gemm_wave_quantization_streamk_pdl_selection",
        "round": 201,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "gpu_sms": gpu_sms,
            "output_tiles": output_tiles,
            "kernels_per_decode_step": kernels_per_decode_step,
            "dp_sm_efficiency_pct": dp_sm_efficiency_pct,
            "streamk_sm_efficiency_pct": 96.8,
            "tpot_speedup_x": round(9.80 / 5.68, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_skinny_gemm_pdl_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

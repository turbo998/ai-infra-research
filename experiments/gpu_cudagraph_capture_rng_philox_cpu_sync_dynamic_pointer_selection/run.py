#!/usr/bin/env python3
"""Round 267: PyTorch CUDAGraph Capture Pitfalls & Silent Bugs (Pointer Rebind vs Frozen Scalar vs Philox RNG vs Static Buffer Contract) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_pytorch_cudagraph_pitfall_tiers() -> dict[str, Any]:
    """Evaluate 4 PyTorch CUDAGraph execution & state management architectures."""
    tiers = [
        {
            "tier_id": "tier1_eager_mode_without_cudagraph_450_kernel_launch_overhead",
            "name": "Tier 1: 纯 Eager 逐算子启动 (每步发射 450 个小算子，CPU Launch 开销 2.70ms 导致 GPU 断流，单步 8.20ms)",
            "step_latency_ms": 8.20,
            "output_correctness_rate_pct": 100.0,
            "rng_entropy_uniqueness_pct": 100.0,
            "verdict": "虽无图捕获 Bug，但每个 Decode/训练步 CPU 发射 450 次 CUDA Kernel 耗时高于 GPU 实际算子执行时间 ✕",
        },
        {
            "tier_id": "tier2_cudagraph_with_python_variable_rebind_and_host_scalar_args",
            "name": "Tier 2: 开启 CUDAGraph 但重绑定 Python 变量 (x = new_x) 或传入 Python int/float 标量 (100% 读旧地址脏数据!)",
            "step_latency_ms": 5.52,
            "output_correctness_rate_pct": 0.0,
            "rng_entropy_uniqueness_pct": 0.0,
            "verdict": "CUDAGraph 焊死了捕获时的 64 位显存虚拟地址与标量字面量! 改 Python 变量指向完全不生效，永远输出第 1 步结果 ✕",
        },
        {
            "tier_id": "tier3_cudagraph_static_copy_but_frozen_triton_host_rng_seed",
            "name": "Tier 3: 使用 static_x.copy_() 原地更新，但自定义 Triton Dropout/采样算子传入了 Python int 随机种子",
            "step_latency_ms": 5.52,
            "output_correctness_rate_pct": 92.5,
            "rng_entropy_uniqueness_pct": 0.0,
            "verdict": "输入数据虽正常更新，但自定义算子的随机种子被永远冻结在捕获那一刻，导致每步采样/Dropout 重复同一伪随机序列 ⚠️",
        },
        {
            "tier_id": "tier4_static_buffer_inplace_copy_plus_device_tensor_metadata_and_philox_rng",
            "name": "Tier 4: 黄金 CUDAGraph 契约 (静态缓冲 .copy_() + 动态标量全转 GPU 0D Tensor + 显存态 Philox RNG 偏移自增)",
            "step_latency_ms": 5.52,
            "output_correctness_rate_pct": 100.0,
            "rng_entropy_uniqueness_pct": 100.0,
            "verdict": "消灭 2.68ms CPU 启动开销 (提速 48.6% 至 5.52ms)，且 100% 动态更新地址、长度元数据与每步独立伪随机数 🏆",
        },
    ]

    return {
        "experiment_id": "gpu_cudagraph_capture_rng_philox_cpu_sync_dynamic_pointer_selection",
        "round": 267,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "kernels_per_step": 450,
            "cpu_launch_overhead_saved_ms": round(8.20 - 5.52, 2),
            "speedup_pct": round((8.20 - 5.52) / 5.52 * 100.0, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_pytorch_cudagraph_pitfall_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

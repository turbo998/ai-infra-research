#!/usr/bin/env python3
"""Round 238: FlashAttention-2/3/4 Backward Pass Non-Determinism (dQ_accum atomicAdd) vs Deterministic Ordered Accumulation Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_flashattention_deterministic_bwd_tiers(
    peak_bwd_tflops: float = 820.0,
) -> dict[str, Any]:
    """Evaluate 4 FlashAttention backward determinism & workspace architectures on Hopper/Blackwell."""
    tiers = [
        {
            "tier_id": "tier1_default_fa2_fa3_non_deterministic_dq_atomicadd",
            "name": "Tier 1: 默认 FlashAttention-2/3 反向传播 (deterministic=False，多 CTA 并发向 dQ_accum 执行乱序 atomicAdd)",
            "bwd_effective_tflops": 820.0,
            "extra_dq_workspace_hbm_gb": 0.25,
            "max_ulp_drift_per_step": 6,
            "bitwise_reproducible": False,
            "verdict": "为了并行切分 K/V 维度，多个 CTA 同时对同一行 dQ_accum 发起硬件 atomicAdd，浮点非结合律导致每步产生 6 ULP 随机漂移 ✕",
        },
        {
            "tier_id": "tier2_legacy_sequential_single_cta_deterministic_fallback",
            "name": "Tier 2: 退回单 CTA 串行扫描 K/V 块消除 atomicAdd (虽实现 0 ULP 确定性，但小 Batch 下 SM 大面积闲置)",
            "bwd_effective_tflops": 310.0,
            "extra_dq_workspace_hbm_gb": 0.25,
            "max_ulp_drift_per_step": 0,
            "bitwise_reproducible": True,
            "verdict": "取消跨 KV 块的 CTA 并行切分会导致头数较少 (GQA/MLA) 时大量 SM 无活可干，反向算力暴跌 62.2% 至 310 TFLOPS ⚠️",
        },
        {
            "tier_id": "tier3_unbounded_per_split_fp32_dq_workspace_reduction",
            "name": "Tier 3: 粗暴分配全量 [NumSplits, B, H, S, D] FP32 私有缓冲区再固定顺序求和 (吞噬 +16.0 GB 显存)",
            "bwd_effective_tflops": 710.0,
            "extra_dq_workspace_hbm_gb": 16.00,
            "max_ulp_drift_per_step": 0,
            "bitwise_reproducible": True,
            "verdict": "虽消除了并发原子写冲突，但在 32K~64K 长序列下多副本 FP32 dQ 缓冲区瞬间吃掉 16 GB 显存 ✅",
        },
        {
            "tier_id": "tier4_fa3_ordered_semaphore_or_sm_bounded_split_dq_deterministic",
            "name": "Tier 4: FA2.6+/FA3 有界 SM 分片缓冲 + 硬件信号量按序累加 (deterministic=True，0 ULP 且仅慢 4.2%)",
            "bwd_effective_tflops": 785.0,
            "extra_dq_workspace_hbm_gb": 0.50,
            "max_ulp_drift_per_step": 0,
            "bitwise_reproducible": True,
            "verdict": "通过按 SM 波次有界分配缓冲区或使用全局信号量强制各 KV 块按 j=0,1,2... 严格保序累加 dQ，仅付 4.2% 代价即实现 0 ULP 逐比特复现 🏆",
        },
    ]

    return {
        "experiment_id": "llm_flashattention_deterministic_backward_splitkv_atomic_workspace_selection",
        "round": 238,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "peak_bwd_tflops": peak_bwd_tflops,
            "deterministic_overhead_pct": round((820.0 - 785.0) / 820.0 * 100.0, 1),
            "speedup_vs_sequential_x": round(785.0 / 310.0, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_flashattention_deterministic_bwd_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

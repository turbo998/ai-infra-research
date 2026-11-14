#!/usr/bin/env python3
"""Round 213: GPU L2 Cache Tile Swizzling (GROUP_SIZE_M=8), Dual-Partition Crossbar & cudaAccessPolicyWindow Persisting Pinning Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_l2_swizzling_persisting_tiers(
    l2_size_mb: int = 50,
) -> dict[str, Any]:
    """Evaluate 4 GPU L2 cache locality and persistence tiers on Hopper H100 (50MB L2) / Blackwell B200 (126MB L2)."""
    tiers = [
        {
            "tier_id": "tier1_naive_row_major_cta_order_l2_thrashing",
            "name": "Tier 1: 朴素行优先 CTA 调度 (pid_m = pid // grid_n · 132 个 SM 同时拉取 64 列 B 矩阵撑爆 L2)",
            "l2_hit_rate_pct": 28.5,
            "hbm_read_amplification_x": 3.4,
            "sustained_tflops": 465.0,
            "verdict": "单波 132 个 CTA 横跨全部 64 列 B Tile，仅 2 个 CTA 共享同一列 B，L2 命中率跌至 28.5%，被 HBM 带宽卡死在 465 TFLOPS ✕",
        },
        {
            "tier_id": "tier2_naive_swizzle_without_streaming_pollution_guard",
            "name": "Tier 2: 粗粒度分块但未隔离一次性流式数据 (大权重流冲刷掉热点 Shared Prefix KV / 共享专家)",
            "l2_hit_rate_pct": 58.0,
            "hbm_read_amplification_x": 1.9,
            "sustained_tflops": 615.0,
            "verdict": "虽提升了单算子内局部性，但跨算子流式读取 140GB 权重会瞬间把 50MB L2 中的高频共享前缀 KV 全部挤出 ⚠️",
        },
        {
            "tier_id": "tier3_2d_grouped_cta_tile_swizzling_group_m_8",
            "name": "Tier 3: 二维分组方块 CTA 调度 (GROUP_SIZE_M=8 · 8x8 超级块仅读 16 个输入块算出 64 个输出块)",
            "l2_hit_rate_pct": 81.2,
            "hbm_read_amplification_x": 1.25,
            "sustained_tflops": 725.0,
            "verdict": "将 64 个 CTA 排列为 8x8 方阵，HBM 矩阵块复用率提升 4 倍，L2 命中率跃升至 81.2%，算力达 725 TFLOPS ✅",
        },
        {
            "tier_id": "tier4_swizzle_plus_l2_persisting_window_and_tma_multicast",
            "name": "Tier 4: 2D Swizzle + cudaAccessPolicyWindow (Persisting 锁定热数据 + Streaming 防污染) + 同半区 TMA Multicast",
            "l2_hit_rate_pct": 89.6,
            "hbm_read_amplification_x": 1.05,
            "sustained_tflops": 792.0,
            "verdict": "用 Persisting 窗口将高频共享前缀/共享专家钉在 5.5 TB/s L2 内，配合流式旁路与 GPC TMA 组播，命中率 89.6%，跑满 792 TFLOPS 🏆",
        },
    ]

    return {
        "experiment_id": "gpu_l2_cache_swizzling_persisting_window_selection",
        "round": 213,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "l2_size_mb": l2_size_mb,
            "l2_bandwidth_tbs": 5.5,
            "speedup_over_row_major_x": round(792.0 / 465.0, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_l2_swizzling_persisting_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

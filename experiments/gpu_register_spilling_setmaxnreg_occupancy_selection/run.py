#!/usr/bin/env python3
"""Round 212: GPU Register Spilling (Local Memory STL/LDL), maxrregcount & Hopper/Blackwell setmaxnreg Warp-Group Reallocation Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_register_setmaxnreg_tiers(
    sm_registers_total: int = 65536,
) -> dict[str, Any]:
    """Evaluate 4 GPU register allocation and warp-group reallocation tiers on H100 (65,536 regs/SM)."""
    tiers = [
        {
            "tier_id": "tier1_unconstrained_large_tile_register_spill",
            "name": "Tier 1: 未受控大 Tile 寄存器溢出 (Consumer 需 232 Regs > 384 线程静态上限 168 · 触发 Local Memory Spill)",
            "producer_regs_per_thread": 168,
            "consumer_regs_per_thread": 168,
            "spilled_bytes_per_thread": 256,
            "sustained_tflops": 265.0,
            "verdict": "384 线程均分 65,536 寄存器最多每线程 168 个，超出的 64 个累加器溢出至片外 HBM Local Memory (STL/LDL)，算力暴跌至 265 TFLOPS ✕",
        },
        {
            "tier_id": "tier2_brute_force_maxrregcount_128_high_occupancy_trap",
            "name": "Tier 2: 盲目强制 --maxrregcount=128 追求高 Occupancy (强压寄存器引发更严重 Spill)",
            "producer_regs_per_thread": 128,
            "consumer_regs_per_thread": 128,
            "spilled_bytes_per_thread": 368,
            "sustained_tflops": 310.0,
            "verdict": "误以为 Occupancy 越高越好，强行压低每线程寄存器上限导致 WGMMA 累加器大量读写 L1/HBM，仅跑出 310 TFLOPS ⚠️",
        },
        {
            "tier_id": "tier3_uniform_launch_bounds_smaller_tile_no_spill",
            "name": "Tier 3: 传统对称 __launch_bounds__(384, 1) 缩小 Tile (全员 168 Regs · Producer 白白浪费 128 Regs)",
            "producer_regs_per_thread": 168,
            "consumer_regs_per_thread": 168,
            "spilled_bytes_per_thread": 0,
            "sustained_tflops": 590.0,
            "verdict": "虽然通过缩小 Tile 消除了 Spill，但 128 个只负责发 TMA 指令的 Producer 线程白白占着 168 个寄存器不用 ✅",
        },
        {
            "tier_id": "tier4_hopper_setmaxnreg_asymmetric_warpgroup_reallocation",
            "name": "Tier 4: Hopper/Blackwell setmaxnreg 动态捐赠 (Producer 退还至 40 Regs -> Consumer 扩容至 232 Regs)",
            "producer_regs_per_thread": 40,
            "consumer_regs_per_thread": 232,
            "spilled_bytes_per_thread": 0,
            "sustained_tflops": 785.0,
            "verdict": "Producer 执行 setmaxnreg.dec 40 捐出 16,384 个寄存器给 Consumer (setmaxnreg.inc 232)，零溢出装下 128x256 巨型累加器，狂飙 785 TFLOPS 🏆",
        },
    ]

    return {
        "experiment_id": "gpu_register_spilling_setmaxnreg_occupancy_selection",
        "round": 212,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "sm_registers_total": sm_registers_total,
            "cta_threads": 384,
            "total_used_regs_tier4": 128 * 40 + 256 * 232,  # 64,512 <= 65,536
            "speedup_over_spill_x": round(785.0 / 265.0, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_register_setmaxnreg_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

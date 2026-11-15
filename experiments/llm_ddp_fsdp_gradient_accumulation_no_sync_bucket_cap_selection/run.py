#!/usr/bin/env python3
"""Round 247: FSDP/FSDP2 Gradient Accumulation (GAS=8) Paradox: Naive 24x Comm Storm vs no_sync() +140GB OOM vs HSDP Selective Sync Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_fsdp_gradient_accumulation_no_sync_tiers(
    gas_steps: int = 8,
) -> dict[str, Any]:
    """Evaluate 4 FSDP / HSDP gradient accumulation (GAS=8) synchronization strategies for a 70B LLM."""
    tiers = [
        {
            "tier_id": "tier1_default_fsdp_every_microbatch_reducescatter_comm_storm",
            "name": "Tier 1: 默认全分片 FSDP 每微批同步 (GAS=8 触发 8×3=24 次跨机全模型集合通信，慢速 RoCE 彻底阻塞)",
            "peak_gradient_hbm_gb": 17.5,
            "inter_node_roce_collectives_per_step": 24,
            "optimizer_step_latency_sec": 14.80,
            "verdict": "未开启 no_sync 时每个微批都要跨机执行 Forward AG + Backward AG + ReduceScatter，8 个微批产生 24 次跨机洪流，耗时 14.80s ✕",
        },
        {
            "tier_id": "tier2_blind_fsdp_no_sync_unsharded_140gb_gradient_oom_crash",
            "name": "Tier 2: 照搬 DDP 在全量 FSDP 上调用 no_sync() (停止 ReduceScatter 迫使每张卡驻留完整 140GB 未切分梯度当场 OOM!)",
            "peak_gradient_hbm_gb": 140.0,
            "inter_node_roce_collectives_per_step": 17,
            "optimizer_step_latency_sec": 99.00,
            "verdict": "FSDP 之所以省梯度显存全靠算完一层立即 ReduceScatter 释放 7/8 梯度；一旦调用 no_sync() 停止切分，140GB 完整梯度瞬间撑爆 80GB H100 ✕",
        },
        {
            "tier_id": "tier3_fsdp2_reshard_after_forward_false_on_intermediate_microbatches",
            "name": "Tier 3: 保持梯度每微批 ReduceScatter，但在微批间设置 reshard_after_forward=False 免除重复 AllGather",
            "peak_gradient_hbm_gb": 17.5,
            "inter_node_roce_collectives_per_step": 9,
            "optimizer_step_latency_sec": 9.40,
            "verdict": "不释放已拉取的前向权重分片从而省去后续 7 个微批的重复 AllGather，同时坚持每微批切分梯度防止 140GB OOM ✅",
        },
        {
            "tier_id": "tier4_hsdp_2d_intra_node_nvlink_shard_plus_inter_node_roce_no_sync",
            "name": "Tier 4: 2D HSDP 分层同步 (机内 NVLink 每微批 ReduceScatter 锁住 17.5GB 显存 + 跨机 RoCE 前 7 批 no_sync 仅末批 AllReduce)",
            "peak_gradient_hbm_gb": 17.5,
            "inter_node_roce_collectives_per_step": 1,
            "optimizer_step_latency_sec": 6.85,
            "verdict": "机内 900GB/s NVLink 每微批切分梯度保住 17.5GB 极小显存，跨机慢速 RoCE 网在前 7 个微批完全静默、仅第 8 批同步 1 次，提速 2.16x 🏆",
        },
    ]

    return {
        "experiment_id": "llm_ddp_fsdp_gradient_accumulation_no_sync_bucket_cap_selection",
        "round": 247,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "gas_steps": gas_steps,
            "step_speedup_x": round(14.80 / 6.85, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_fsdp_gradient_accumulation_no_sync_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

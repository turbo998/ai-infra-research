#!/usr/bin/env python3
"""Round 204: LLM Serving Sleep Mode (vLLM / SGLang CuMemAllocator), CUDA Graph Pointer Preservation & RL Co-Location Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_sleep_mode_tiers(
    hbm_capacity_gb: float = 80.0,
) -> dict[str, Any]:
    """Evaluate 4 GPU memory switching tiers for 70B TP=4 multi-model hot-swap and RLHF/GRPO co-location."""
    tiers = [
        {
            "tier_id": "tier1_process_kill_and_cold_restart",
            "name": "Tier 1: 销毁进程并冷重启 (重建 NCCL + 重载权重 + 重捕获 50 张 CUDA Graphs)",
            "sleep_latency_s": 2.5,
            "wake_up_latency_s": 65.0,
            "residual_gpu_hbm_gb": 0.0,
            "cudagraph_preserved": False,
            "verdict": "每次切换都要花 65 秒重新初始化 NCCL、从磁盘读权重并重新捕获 50 张 CUDA Graph，RL 每步训练切换耗掉一半时间 ✕",
        },
        {
            "tier_id": "tier2_standard_cudafree_empty_cache_recapture",
            "name": "Tier 2: 传统 cudaFree / empty_cache (虚拟地址失效迫使重捕获 CUDA Graph)",
            "sleep_latency_s": 1.1,
            "wake_up_latency_s": 38.0,
            "residual_gpu_hbm_gb": 3.8,
            "cudagraph_preserved": False,
            "verdict": "cudaFree 会销毁原虚拟内存地址，导致已捕获的 CUDA Graph 指针全部失效，唤醒时仍需 38 秒重录计算图 ⚠️",
        },
        {
            "tier_id": "tier3_sleep_level1_cumem_cpu_pinned_offload",
            "name": "Tier 3: Sleep Level 1 (CuMemAllocator 保留虚拟地址 + 权重卸载至 Host Pinned DRAM)",
            "sleep_latency_s": 0.72,
            "wake_up_latency_s": 0.68,
            "residual_gpu_hbm_gb": 0.6,
            "cudagraph_preserved": True,
            "verdict": "利用 cuMemUnmap 释放 99.2% 物理显存但保留原虚拟指针，0.68 秒通过 PCIe Gen5 拷回权重直接复跑原 CUDA Graph (快 95x) ✅",
        },
        {
            "tier_id": "tier4_sleep_level2_cumem_discard_plus_ipc_refit",
            "name": "Tier 4: Sleep Level 2 (直接丢弃物理页 + 唤醒后原地重映射接收 Trainer CUDA IPC 新权重)",
            "sleep_latency_s": 0.08,
            "wake_up_latency_s": 0.22,
            "residual_gpu_hbm_gb": 0.6,
            "cudagraph_preserved": True,
            "verdict": "专为同卡 RL (veRL/OpenRLHF) 打造：0.08s 极速释放 79.4GB 显存，0.22s 原地重映射接收新权重，切换提速 295 倍 🏆",
        },
    ]

    return {
        "experiment_id": "llm_serving_sleep_mode_cumem_rl_colocation_selection",
        "round": 204,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "hbm_capacity_gb": hbm_capacity_gb,
            "freed_hbm_pct": 99.2,
            "level1_speedup_x": round(65.0 / 0.68, 1),
            "level2_speedup_x": round(65.0 / 0.22, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_sleep_mode_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

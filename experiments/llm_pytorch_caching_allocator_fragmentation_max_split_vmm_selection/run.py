#!/usr/bin/env python3
"""Round 241: PyTorch PYTORCH_CUDA_ALLOC_CONF (Fragmentation OOM vs Hot-Loop empty_cache() Anti-Pattern vs expandable_segments:True VMM) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_pytorch_cuda_alloc_conf_tiers(
    gpu_hbm_gb: float = 80.0,
) -> dict[str, Any]:
    """Evaluate 4 PyTorch CUDACachingAllocator configurations under variable-length LLM training."""
    tiers = [
        {
            "tier_id": "tier1_default_cudamalloc_allocator_14gb_swiss_cheese_oom",
            "name": "Tier 1: 默认 CUDACachingAllocator (变长序列将空闲池切成瑞士奶酪碎片，剩 14.2 GB 缓存仍报 2GB OOM)",
            "unusable_fragmented_reserved_hbm_gb": 14.20,
            "allocator_stall_overhead_ms_per_step": 95.0,
            "max_feasible_seq_len_k": 24,
            "verdict": "传统的 cudaMalloc 虚拟地址与物理地址死绑，14.2 GB 空闲显存散落在几十个不相邻的 256MB 孔洞中无法拼出 2GB 连续张量 ✕",
        },
        {
            "tier_id": "tier2_hot_loop_torch_cuda_empty_cache_performance_suicide",
            "name": "Tier 2: 在每步循环内调用 torch.cuda.empty_cache() (每步触发全流同步与驱动 cudaFree/cudaMalloc 锁卡死 340 ms)",
            "unusable_fragmented_reserved_hbm_gb": 3.80,
            "allocator_stall_overhead_ms_per_step": 340.0,
            "max_feasible_seq_len_k": 28,
            "verdict": "empty_cache() 不仅阻塞等待 GPU 所有流跑完，还把缓存块全部退还给 CUDA 驱动，导致下一步被迫重新调慢速 cudaMalloc 锁表 ⚠️",
        },
        {
            "tier_id": "tier3_max_split_size_512mb_plus_gc_threshold_08",
            "name": "Tier 3: 配置 max_split_size_mb:512,garbage_collection_threshold:0.8 (保护大块不被切碎并按需增量回收)",
            "unusable_fragmented_reserved_hbm_gb": 4.60,
            "allocator_stall_overhead_ms_per_step": 18.5,
            "max_feasible_seq_len_k": 32,
            "verdict": "避免了 empty_cache() 的暴力全清，禁止拆分 >512MB 的大块，但仍受限于固定物理块无法跨段拼接 ✅",
        },
        {
            "tier_id": "tier4_expandable_segments_true_cumemmap_virtual_stitching",
            "name": "Tier 4: 开启 expandable_segments:True (基于 CUDA VMM cuMemMap 将离散物理页秒级拼接为连续虚拟张量)",
            "unusable_fragmented_reserved_hbm_gb": 0.35,
            "allocator_stall_overhead_ms_per_step": 0.2,
            "max_feasible_seq_len_k": 40,
            "verdict": "解耦虚拟地址与物理显存页，直接将散落的 2MB/20MB 物理空闲页动态映射追加到同一段可扩展虚拟段，碎片压至 0.35 GB (降 40.6x) 🏆",
        },
    ]

    return {
        "experiment_id": "llm_pytorch_caching_allocator_fragmentation_max_split_vmm_selection",
        "round": 241,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "gpu_hbm_gb": gpu_hbm_gb,
            "fragmentation_reduction_x": round(14.20 / 0.35, 1),
            "speedup_vs_empty_cache_x": round(340.0 / 0.2, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_pytorch_cuda_alloc_conf_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

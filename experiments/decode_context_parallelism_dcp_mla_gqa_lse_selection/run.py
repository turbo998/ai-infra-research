#!/usr/bin/env python3
"""Round 216: Decode Context Parallelism (DCP) for DeepSeek-V3 MLA & GQA Long-Context KV Sequence Sharding Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_dcp_mla_tiers(
    context_tokens: int = 131072,
    num_gpus: int = 8,
) -> dict[str, Any]:
    """Evaluate 4 KV sharding tiers for 128K-context Decode on DeepSeek-V3 MLA (1 shared latent KV head) across 8 GPUs."""
    tiers = [
        {
            "tier_id": "tier1_standard_tp8_replicated_mla_kv_cache",
            "name": "Tier 1: 传统纯 TP=8 权重切分 (MLA 仅 1 个共享潜变量 KV 头无法按头切分 · 8 卡各存 100% 完整 KV)",
            "kv_replication_factor_x": 8,
            "per_gpu_kv_tokens_read": 131072,
            "decode_step_tbt_ms": 28.4,
            "max_concurrent_128k_reqs": 8,
            "verdict": "DeepSeek-V3 MLA 只有 1 个共享压缩 KV 头 (c_KV)，纯 TP=8 下每张卡都得全量复制并重读 128K KV，显存与带宽浪费 8 倍 ✕",
        },
        {
            "tier_id": "tier2_contiguous_chunk_dcp8_prefill_imbalance",
            "name": "Tier 2: 粗粒度连续分块 DCP=8 (GPU r 存 [r*16K, (r+1)*16K] · 短请求负载集中在 Rank 0)",
            "kv_replication_factor_x": 1,
            "per_gpu_kv_tokens_read": 16384,
            "decode_step_tbt_ms": 8.9,
            "max_concurrent_128k_reqs": 56,
            "verdict": "虽然消除了 8 倍 KV 复制，但按连续 16K 区间切分会导致变长请求批次中低号 GPU 过载、高号 GPU 空闲 ⚠️",
        },
        {
            "tier_id": "tier3_interleaved_block_dcp8_allgather_lse",
            "name": "Tier 3: 细粒度 Page-Block 轮询交错 DCP=8 + 标准 NCCL AllGather (O_local, LSE_local)",
            "kv_replication_factor_x": 1,
            "per_gpu_kv_tokens_read": 16384,
            "decode_step_tbt_ms": 5.4,
            "max_concurrent_128k_reqs": 64,
            "verdict": "按 64-Token Page Block 在 8 张卡间 Round-Robin 交错分布，任意长度请求 8 卡绝对均衡，并发容量暴涨 8 倍 ✅",
        },
        {
            "tier_id": "tier4_hybrid_tp_weights_dcp_kv_fused_nvlink_lse_reduce",
            "name": "Tier 4: 混合 TP=8 权重 + Interleaved DCP=8 序列切分 + NVLink 融合 LSE Reduce-Scatter (vLLM/SGLang/Helix)",
            "kv_replication_factor_x": 1,
            "per_gpu_kv_tokens_read": 16384,
            "decode_step_tbt_ms": 4.2,
            "max_concurrent_128k_reqs": 64,
            "verdict": "每张卡仅存读 1/8 序列长度 (16K Tokens)，通过 NVLink 融合 LSE 归并与 O_proj Reduce-Scatter，TBT 压至 4.2 ms (快 6.76x) 🏆",
        },
    ]

    return {
        "experiment_id": "decode_context_parallelism_dcp_mla_gqa_lse_selection",
        "round": 216,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "context_tokens": context_tokens,
            "num_gpus": num_gpus,
            "kv_memory_saving_x": 8.0,
            "tbt_speedup_x": round(28.4 / 4.2, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_dcp_mla_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Round 221: PyTorch FSDP1 (FlatParameter & recordStream Bloat) vs. FSDP2 (DTensor Per-Parameter Sharding) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_fsdp2_dtensor_recordstream_tiers(
    gpu_hbm_capacity_gb: float = 80.0,
) -> dict[str, Any]:
    """Evaluate 4 FSDP sharding & stream-memory management tiers on 64x H100 (80GB) training a 70B LLM."""
    tiers = [
        {
            "tier_id": "tier1_fsdp1_flat_parameter_record_stream_allocator_bloat",
            "name": "Tier 1: 传统 FSDP1 (FlatParameter 一维拼接 + record_stream 延迟释放)",
            "peak_reserved_hbm_gb": 79.4,
            "record_stream_inflight_bloat_gb": 14.8,
            "cuda_malloc_retry_stalls_per_100_steps": 18,
            "step_throughput_tokens_per_sec_per_gpu": 3820,
            "verdict": "CPU 调度远超 GPU 计算流，导致 record_stream 锁死多个 Transformer 层的 AllGather 缓冲区，引爆 +14.8 GB 显存膨胀与 18 次分配器清缓存卡顿 ✕",
        },
        {
            "tier_id": "tier2_fsdp1_limit_all_gathers_cpu_sync_brake",
            "name": "Tier 2: FSDP1 + limit_all_gathers=True (靠强制阻塞 CPU 抑制显存暴涨，但无法混用冻结/Float8)",
            "peak_reserved_hbm_gb": 65.2,
            "record_stream_inflight_bloat_gb": 2.4,
            "cuda_malloc_retry_stalls_per_100_steps": 2,
            "step_throughput_tokens_per_sec_per_gpu": 4180,
            "verdict": "虽通过阻塞 CPU 线程将显存压回 65.2 GB，但一维 FlatParameter 仍把冻结权重与可训练权重强行绑死，且无法与 Float8Tensor/TP 无缝组合 ⚠️",
        },
        {
            "tier_id": "tier3_fsdp2_fully_shard_dtensor_deterministic_free",
            "name": "Tier 3: PyTorch FSDP2 (fully_shard 逐参数 DTensor + 零 record_stream 确定性双流释放)",
            "peak_reserved_hbm_gb": 62.4,
            "record_stream_inflight_bloat_gb": 0.0,
            "cuda_malloc_retry_stalls_per_100_steps": 0,
            "step_throughput_tokens_per_sec_per_gpu": 4460,
            "verdict": "每个参数独立按 dim-0 切分为 DTensor，彻底消灭 record_stream（0.0 GB 膨胀、0 次重试），原生支持逐参数冻结、LoRA 与 DCP 分布式保存 ✅",
        },
        {
            "tier_id": "tier4_fsdp2_dtensor_2d_tp_hsdp_float8_composable",
            "name": "Tier 4: FSDP2 (fully_shard) + 2D DeviceMesh (TP/HSDP) + Float8 AllGather 通信压缩",
            "peak_reserved_hbm_gb": 60.8,
            "record_stream_inflight_bloat_gb": 0.0,
            "cuda_malloc_retry_stalls_per_100_steps": 0,
            "step_throughput_tokens_per_sec_per_gpu": 4940,
            "verdict": "利用 DTensor 子类组合性在 AllGather 前先转为 Float8（跨机通信量减半）并与机内 TP / 跨机 HSDP 无缝复合，显存压至 60.8 GB、吞吐提速 1.29x 🏆",
        },
    ]

    return {
        "experiment_id": "pytorch_fsdp2_dtensor_per_param_recordstream_selection",
        "round": 221,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "gpu_hbm_capacity_gb": gpu_hbm_capacity_gb,
            "hbm_saved_gb": round(79.4 - 60.8, 1),
            "throughput_speedup_x": round(4940 / 3820, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_fsdp2_dtensor_recordstream_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

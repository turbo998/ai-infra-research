#!/usr/bin/env python3
"""Round 242: Prefill-Decode (PD) Disaggregation KV Transfer (Monolithic Blind Push vs Host Bounce vs Layer-by-Layer Pipelined RDMA) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_pd_disagg_layerwise_rdma_tiers(
    kv_payload_gb: float = 10.2,
) -> dict[str, Any]:
    """Evaluate 4 PD-disaggregated KV Cache transfer & backpressure architectures for a 32K prompt."""
    tiers = [
        {
            "tier_id": "tier1_monolithic_end_of_prefill_blind_push_stall_and_oom",
            "name": "Tier 1: 全层算完后整块盲推 (等 80 层全部算完才启动 RDMA，暴露 245ms 传输泡沫且无 Decode 预留致 14.5% 爆显存)",
            "exposed_kv_transfer_latency_ms": 245.0,
            "decode_oom_or_drop_rate_pct": 14.5,
            "layerwise_compute_transfer_overlap_pct": 0.0,
            "verdict": "等第 80 层算完才开始传 10.2 GB KV Cache，不仅把 245ms 传输时间 100% 暴露在关键路径上，且盲推会直接冲垮 Decode 节点剩余显存 ✕",
        },
        {
            "tier_id": "tier2_host_dram_bounce_buffer_two_hop_staging",
            "name": "Tier 2: 经由主机内存中转池 (Prefill HBM -> Host DRAM -> RDMA -> Decode DRAM -> HBM，防止 OOM 但暴露 380ms)",
            "exposed_kv_transfer_latency_ms": 380.0,
            "decode_oom_or_drop_rate_pct": 0.0,
            "layerwise_compute_transfer_overlap_pct": 0.0,
            "verdict": "虽利用主机大容量内存池缓冲消除了 Decode 端 OOM，但多出两次 PCIe D2H/H2D 搬运，暴露延迟高达 380ms ⚠️",
        },
        {
            "tier_id": "tier3_preallocated_slot_end_of_prefill_gpudirect_rdma",
            "name": "Tier 3: 预分配槽位 + 全层算完后 GPUDirect RDMA 直拉 (0% OOM，但仍未与逐层计算重叠，暴露 210ms)",
            "exposed_kv_transfer_latency_ms": 210.0,
            "decode_oom_or_drop_rate_pct": 0.0,
            "layerwise_compute_transfer_overlap_pct": 0.0,
            "verdict": "由 Decode 提前分配物理块地址再走 GPUDirect RDMA 零拷贝拉取，彻底消除 OOM，但仍白白闲置了整个 Prefill 期间的网卡带宽 ✅",
        },
        {
            "tier_id": "tier4_preallocated_slot_plus_layer_by_layer_pipelined_gpudirect_rdma",
            "name": "Tier 4: 预分配槽位握手 + 逐层异步重叠流水线 (算第 L+1 层时同步传第 L 层 KV，仅暴露末层 3.2ms!)",
            "exposed_kv_transfer_latency_ms": 3.2,
            "decode_oom_or_drop_rate_pct": 0.0,
            "layerwise_compute_transfer_overlap_pct": 98.7,
            "verdict": "Prefill 刚算完第 L 层即触发 Event 将该层 KV 直写至 Decode 预留显存槽，当第 80 层算完时前 79 层早已到齐，暴露延迟骤降 76.6 倍至 3.2ms 🏆",
        },
    ]

    return {
        "experiment_id": "llm_pd_disaggregation_kv_transfer_layerwise_pipelining_selection",
        "round": 242,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "kv_payload_gb": kv_payload_gb,
            "exposed_latency_reduction_x": round(245.0 / 3.2, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_pd_disagg_layerwise_rdma_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

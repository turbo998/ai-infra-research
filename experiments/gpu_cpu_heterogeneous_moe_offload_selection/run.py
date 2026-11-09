from __future__ import annotations

import json
from pathlib import Path


def evaluate_heterogeneous_offload_tiers() -> dict:
    """Deterministic evaluator for 4 GPU-CPU Heterogeneous Offloading Tiers on 671B MoE.

    Workload Baseline:
      - Model: DeepSeek-R1 / V3 671B MoE (Q4_K_M / INT4 = ~380 GB Total Weights; 37B Active per Token)
      - Single-Server Hardware: 1×GPU (48 GB VRAM) + Dual-Socket Xeon/EPYC Host (512 GB 24-Channel DDR5 @ 760 GB/s, PCIe Gen5 x16 @ 52 GB/s)
      - Compares:
        1) Interactive Single-User Decode Speed (tokens / sec at Batch=1)
        2) Offline Batch=128 Throughput (tokens / sec)
        3) PCIe Bus Bytes Transferred per Token (MB)
        4) 16K Context Prefill TTFT (seconds)
    """
    tiers = [
        {
            "tier_id": "tier1_cuda_unified_memory_page_fault_swap",
            "name": "Tier 1: Naive OS Swap / CUDA Unified Memory (cudaMallocManaged On-Demand Page Faults)",
            "name_zh": "Tier 1: 原生 OS Swap / CUDA Unified Memory 缺页中断 (按需 4KB/64KB 缺页拉取)",
            "execution_paradigm": "GPU faults on missing expert pages -> PCIe 64KB thrashing -> GPU computes",
            "interactive_batch1_decode_tps": 0.35,
            "offline_batch128_throughput_tps": 4.2,
            "pcie_bytes_transferred_per_token_mb": 21500.0,
            "prefill_16k_ttft_sec": 185.0,
            "hardware_capex_usd": 18500,
        },
        {
            "tier_id": "tier2_layerwise_weight_dma_offload_accelerate",
            "name": "Tier 2: Layer-Wise Weight DMA Offloading (HuggingFace accelerate device_map / Pinned H2D Copy)",
            "name_zh": "Tier 2: 按层权重 DMA 搬运卸载 (把 17GB 激活专家权重每 Token 从内存拷进 GPU 计算)",
            "execution_paradigm": "Pinned RAM -> Async cudaMemcpyAsync of 17 GB active weights over PCIe (52 GB/s) -> GPU GEMM",
            "interactive_batch1_decode_tps": 2.10,
            "offline_batch128_throughput_tps": 48.0,
            "pcie_bytes_transferred_per_token_mb": 17000.0,
            "prefill_16k_ttft_sec": 42.0,
            "hardware_capex_usd": 18500,
        },
        {
            "tier_id": "tier3_flexgen_zigzag_block_schedule_offload",
            "name": "Tier 3: Throughput-Oriented Zig-Zag Block Offload (FlexGen / ZeRO-Infinity Column-Major Sweep)",
            "name_zh": "Tier 3: 吞吐导向 Zig-Zag 列优先批量卸载 (FlexGen / ZeRO-Infinity 离线大批次平摊权重搬运)",
            "execution_paradigm": "Loads 1 layer weight once and sweeps across N=128 large batches before advancing to next layer",
            "interactive_batch1_decode_tps": 2.15,
            "offline_batch128_throughput_tps": 310.0,
            "pcie_bytes_transferred_per_token_mb": 265.0,
            "prefill_16k_ttft_sec": 28.0,
            "hardware_capex_usd": 18500,
        },
        {
            "tier_id": "tier4_ktransformers_fiddler_cpu_amx_in_place_moe",
            "name": "Tier 4: Heterogeneous Compute-In-Place (KTransformers / Fiddler: GPU MLA+Hot Experts + CPU AMX/DDR5 Cold Experts)",
            "name_zh": "Tier 4: 异构算子原位协同 (KTransformers / Fiddler: GPU 跑 MLA 与热专家 + 双路 CPU AMX 在 DDR5 原位算冷专家)",
            "execution_paradigm": "Weights NEVER cross PCIe! GPU sends 32 KB activation vector -> Dual-Socket CPU AMX computes experts at 760 GB/s DDR5",
            "interactive_batch1_decode_tps": 18.60,
            "offline_batch128_throughput_tps": 195.0,
            "pcie_bytes_transferred_per_token_mb": 0.032,
            "prefill_16k_ttft_sec": 6.8,
            "hardware_capex_usd": 18500,
        },
    ]

    t2 = tiers[1]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier4_interactive_decode_tps": t4["interactive_batch1_decode_tps"],
        "tier4_vs_tier2_interactive_speedup_x": round(
            t4["interactive_batch1_decode_tps"] / t2["interactive_batch1_decode_tps"], 2
        ),
        "tier3_vs_tier2_offline_batch_speedup_x": round(
            t3["offline_batch128_throughput_tps"] / t2["offline_batch128_throughput_tps"], 2
        ),
        "tier4_pcie_payload_reduction_x": round(
            t2["pcie_bytes_transferred_per_token_mb"] / t4["pcie_bytes_transferred_per_token_mb"], 1
        ),
    }

    return {
        "experiment_id": "gpu_cpu_heterogeneous_moe_offload_selection",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_gpu_cpu_heterogeneous_moe_offload_and_pcie_vs_ddr5_bandwidth_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "KTransformers (Tsinghua MADSys) & Fiddler GPU-CPU Heterogeneous MoE Execution Model",
                "FlexGen (ICML) Zig-Zag Block Schedule & DeepSpeed ZeRO-Infinity Offload Bandwidth Amortization",
                "PCIe Gen5 x16 (52 GB/s) Weight Transfer vs Dual-Socket 24-Channel DDR5 (760 GB/s) + Intel AMX In-Place Compute",
            ],
        },
        "workload_baseline": {
            "model": "DeepSeek-R1 671B MoE (380 GB Q4_K_M, 37B Active per Token)",
            "server_spec": "1x 48GB GPU + Dual-Socket Xeon/EPYC (512 GB DDR5 @ 760 GB/s) — $18,500 Total CapEx",
        },
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_heterogeneous_offload_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

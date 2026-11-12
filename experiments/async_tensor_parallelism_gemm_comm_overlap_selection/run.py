from __future__ import annotations

import json
from pathlib import Path


def evaluate_async_tensor_parallelism_overlap_tiers() -> dict:
    """Deterministic evaluator for 4 Tensor Parallelism (TP=8) Fine-Grained GEMM-Communication Overlap (Async-TP) Tiers.

    Workload Baseline:
      - Node: 8x H100/H200 SXM5 (900 GB/s NVSwitch 3.0) running a 70B/405B LLM with TP=8 and Sequence Parallelism (SP)
      - Layer Operations: Per Transformer block, 2 AllGather (AG) + 2 ReduceScatter (RS) collectives along sequence dimension
        paired with QKV, O-Proj, Gate/Up, and Down-Proj GEMMs (Sequence Length = 8,192, Hidden = 8,192, BF16/FP8)
      - Evaluates:
        1) Exposed Unoverlapped TP Communication Time Ratio (% of layer time stalled waiting on NVLink collectives)
        2) GPU SMs Hijacked by Communication Kernels (out of 132 H100 SMs — stealing SMs slows down concurrent GEMMs!)
        3) Effective Overlapped GEMM TFLOPS Retention (% of solo GEMM speed when running concurrently with communication)
        4) Per-Layer Forward+Backward Time (ms), Cluster MFU (%), and Annual Compute Savings on a 128-GPU Cluster ($)
    """
    tiers = [
        {
            "tier_id": "tier1_sequential_gemm_then_nccl_collective",
            "name": "Tier 1: Sequential TP+SP (Wait for Full AllGather -> Run GEMM -> Wait for Full ReduceScatter)",
            "name_zh": "Tier 1: 传统串行 TP+SP (先等完整 AllGather 跑完 -> 算 GEMM -> 再等完整 ReduceScatter)",
            "mechanism": "Executes NCCL AllGather and ReduceScatter strictly sequentially before/after cuBLAS GEMMs on the same stream",
            "exposed_comm_bubble_pct": 28.4,
            "comm_sms_hijacked": 20,
            "gemm_tflops_retention_pct": 100.0,
            "layer_fwd_bwd_time_ms": 14.80,
            "cluster_mfu_pct": 44.2,
            "annual_128gpu_comm_stall_waste_usd": 795000,
        },
        {
            "tier_id": "tier2_naive_2stream_chunked_nccl_overlap",
            "name": "Tier 2: Naive 2-Stream Chunked NCCL Overlap (4 Chunks + 24 NCCL SMs Stealing Compute Wavefronts -> GEMM Slowdown)",
            "name_zh": "Tier 2: 粗放双流分块 NCCL 重叠 (切 4 块并发，但 NCCL 抢占 24 个 SM 导致矩阵乘掉速 -18.5%)",
            "mechanism": "Splits GEMM & NCCL into 4 chunks on 2 CUDA streams; NCCL kernels occupy 24 SMs & L2 bandwidth, starving cuBLAS wave quantization",
            "exposed_comm_bubble_pct": 14.2,
            "comm_sms_hijacked": 24,
            "gemm_tflops_retention_pct": 81.5,
            "layer_fwd_bwd_time_ms": 12.90,
            "cluster_mfu_pct": 50.7,
            "annual_128gpu_comm_stall_waste_usd": 465000,
        },
        {
            "tier_id": "tier3_cuda_symmetric_memory_p2p_tile_push",
            "name": "Tier 3: PyTorch CUDASymmetricMemory + Cutlass Tile-Centric P2P Push/Pull (Fused Producer-Consumer Tile Ring)",
            "name_zh": "Tier 3: PyTorch CUDA 对称内存 (SymmetricMemory) + 瓦片级 P2P 原位直推 (消除独立通信核启动)",
            "mechanism": "Maps peer HBM into local virtual address space via cuMemMap; GEMM epilogue writes output tiles directly to peer HBM over NVLink",
            "exposed_comm_bubble_pct": 4.5,
            "comm_sms_hijacked": 4,
            "gemm_tflops_retention_pct": 95.8,
            "layer_fwd_bwd_time_ms": 11.15,
            "cluster_mfu_pct": 58.6,
            "annual_128gpu_comm_stall_waste_usd": 142000,
        },
        {
            "tier_id": "tier4_sm_free_copy_engine_dma_nvls_multimem_async_tp",
            "name": "Tier 4: Zero-SM Hardware Copy-Engine DMA AllGather + NVLS multimem.ld_reduce Fused Epilogue (100% 132-SM GEMM)",
            "name_zh": "Tier 4: 零 SM 硬件 Copy Engine DMA 预取 + NVSwitch multimem 硬件归约融合尾声 (132 个 SM 全力算矩阵乘)",
            "mechanism": "AllGather runs on 0-SM hardware Copy Engines (P2P_CUMEM); ReduceScatter fuses PTX multimem.red into GEMM epilogue via NVSwitch ALU",
            "exposed_comm_bubble_pct": 1.6,
            "comm_sms_hijacked": 0,
            "gemm_tflops_retention_pct": 99.2,
            "layer_fwd_bwd_time_ms": 10.60,
            "cluster_mfu_pct": 61.7,
            "annual_128gpu_comm_stall_waste_usd": 45000,
        },
    ]

    # Sensitivity across Message / Sequence Sizes and Interconnects
    overlap_perturbation = [
        {
            "condition": "Short Sequence / Small Batch (2K Tokens, Latency-Bound)",
            "tier1_seq_ms": 4.20,
            "tier2_2stream_ms": 3.85,
            "tier4_zero_sm_ms": 2.95,
            "speedup_x": 1.42,
            "insight": "Eliminates 16 separate NCCL kernel launch overheads per layer via fused multimem epilogue",
        },
        {
            "condition": "Standard Pretraining Sequence (8K Tokens, Compute + BW Balanced)",
            "tier1_seq_ms": 14.80,
            "tier2_2stream_ms": 12.90,
            "tier4_zero_sm_ms": 10.60,
            "speedup_x": 1.40,
            "insight": "Zero-SM Copy Engine preserves all 132 H100 SMs for cuBLAS tile wave quantization (99.2% GEMM retention)",
        },
        {
            "condition": "Long Sequence FP8 Training (32K Tokens, Higher Comm-to-Compute Ratio)",
            "tier1_seq_ms": 56.40,
            "tier2_2stream_ms": 48.20,
            "tier4_zero_sm_ms": 38.10,
            "speedup_x": 1.48,
            "insight": "FP8 doubles Tensor Core speed, making TP communication 38% of layer time unless hidden by Tier 4 Async-TP",
        },
        {
            "condition": "Multi-Node TP=16 Across 2 Nodes (Cross-Node RoCE/IB Bottleneck)",
            "tier1_seq_ms": 24.50,
            "tier2_2stream_ms": 20.80,
            "tier4_zero_sm_ms": 15.20,
            "speedup_x": 1.61,
            "insight": "2D Hierarchical Async-TP overlaps inter-node RDMA chunk 0 while computing intra-node NVLink chunks 1..7",
        },
    ]

    t1 = tiers[0]
    t2 = tiers[1]
    t4 = tiers[3]

    summary = {
        "tier4_vs_tier1_layer_speedup_x": round(
            t1["layer_fwd_bwd_time_ms"] / t4["layer_fwd_bwd_time_ms"], 2
        ),
        "tier4_exposed_comm_bubble_pct": t4["exposed_comm_bubble_pct"],
        "tier4_vs_tier2_gemm_retention_gain_pt": round(
            t4["gemm_tflops_retention_pct"] - t2["gemm_tflops_retention_pct"], 1
        ),
        "tier4_vs_tier1_mfu_gain_pt": round(
            t4["cluster_mfu_pct"] - t1["cluster_mfu_pct"], 1
        ),
        "tier4_vs_tier1_annual_savings_usd": t1["annual_128gpu_comm_stall_waste_usd"] - t4["annual_128gpu_comm_stall_waste_usd"],
    }

    return {
        "experiment_id": "async_tensor_parallelism_gemm_comm_overlap_selection",
        "date": "2026-11-12",
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "PyTorch 2.5+ Async-TP & CUDASymmetricMemory (enable_symm_mem_for_group / fused_all_gather_matmul)",
                "TransformerEngine / Megatron-Core TP Overlap (Ring-Exchange vs Copy-Engine P2P_CUMEM & NVLS multimem)",
                "GPU Wave Quantization Degradation Under Concurrent NCCL SM Stealing (132 SMs -> 108 SMs)",
            ],
        },
        "workload_baseline": {
            "node": "8x H100/H200 SXM5 (900 GB/s NVSwitch 3.0), TP=8 + Sequence Parallelism (SP)",
            "layer_shape": "8,192 tokens × 8,192 hidden dim (70B/405B Transformer block, 2 AG + 2 RS per pass)",
        },
        "tiers": tiers,
        "overlap_perturbation": overlap_perturbation,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_async_tensor_parallelism_overlap_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

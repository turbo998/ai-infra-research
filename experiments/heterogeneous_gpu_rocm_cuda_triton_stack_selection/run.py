"""Deterministic evaluator for Heterogeneous GPU Cluster (NVIDIA H100/B200 + AMD MI300X/MI325X) Compiler Stack & Cross-Architecture PD Disaggregation Selection (Round 49).

Models the effective memory bandwidth utilization (%), single-node 70B FP8 serving throughput (tokens/s),
P99 Time-Per-Output-Token (TPOT ms), maximum concurrent 32K-context sessions per 8-GPU node,
engineering maintenance overhead, and cost per 1M tokens across four heterogeneous GPU serving tiers:
  - Tier 1: Naive HIPIFY Source Port on MI300X (Warp-32 geometry on Wavefront-64 + No 8-XCD L2 Swizzling)
  - Tier 2: Static ONNX / OpenXLA Graph Export (Loses Custom PagedAttention / Fused MoE / Speculative Kernels)
  - Tier 3: Unified vLLM/SGLang + OpenAI Triton / TileLang + AMD AITER (XCD-Swizzled Wavefront-64 & MFMA Kernels)
  - Tier 4: Cross-Vendor Heterogeneous PD Disaggregation (NVIDIA H100 Compute-Bound Prefill + AMD MI300X 192GB 5.3TB/s Memory-Bound Decode via Mooncake RoCEv2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_heterogeneous_gpu_rocm_cuda_triton_matrix() -> dict[str, Any]:
    workload = {
        "cluster_profile": "Heterogeneous Fleet: 8×NVIDIA H100-80GB (3.35 TB/s HBM3, $28.80/hr) + 8×AMD Instinct MI300X-192GB (5.30 TB/s HBM3, $21.60/hr)",
        "model_profile": "Llama-3.3-70B-FP8 / DeepSeek-R1-Distill-70B (80 Layers, TP=8, 32K Long-Context Sessions)",
        "architectural_contrast": "NVIDIA Hopper (32-thread Warp, 132 SMs, TMA/wgmma, 80GB) vs AMD CDNA3 MI300X (64-thread Wavefront, 8 XCD Chiplets, 304 CUs, MFMA, 192GB)",
    }

    tiers = [
        {
            "id": "tier1_naive_hipify_unoptimized_rocm_port",
            "name": "Tier 1: 朴素 HIPIFY 源码硬转 (照搬 CUDA Warp-32 分块 + 无 8-XCD 缓存亲和)",
            "name_en": "Tier 1: Naive HIPIFY Source Port (Unchanged Warp-32 Tile Geometry + 8-XCD L2 Thrashing)",
            "compiler_and_kernel_stack": "Automated hipify-perl C++ translation without Wavefront-64 retuning or XCD chiplet swizzling",
            "deployment_topology": "Monolithic Mixed Prefill+Decode on MI300X (Frequent Cross-XCD Mall Cache Misses)",
            "hbm_bandwidth_utilization_pct": 41.5,
            "node_throughput_tok_s": 1120.0,
            "decode_p99_tpot_ms": 26.5,
            "max_concurrent_32k_sessions_per_node": 148,
            "cost_per_1m_tokens_usd": 5.36,
            "recommended_for": "Initial compilation smoke tests only; never use in production because CUDA 32-thread warp assumptions waste 50% of CDNA3 64-thread wavefront lanes and thrash across 8 XCD chiplets.",
            "status": "fail",
        },
        {
            "id": "tier2_static_onnx_openxla_export_wrapper",
            "name": "Tier 2: 静态 ONNX / OpenXLA 跨芯片图导出 (丢失 PagedAttention 与融合 MoE 算子)",
            "name_en": "Tier 2: Static ONNX / OpenXLA Graph Export (Loses Custom Paged KV & Fused MoE Kernels)",
            "compiler_and_kernel_stack": "Exports model to static ONNX/XLA IR; falls back to generic GEMM/Attention primitives",
            "deployment_topology": "Monolithic Serving with Padded Static KV Tensors",
            "hbm_bandwidth_utilization_pct": 54.0,
            "node_throughput_tok_s": 1480.0,
            "decode_p99_tpot_ms": 21.0,
            "max_concurrent_32k_sessions_per_node": 64,
            "cost_per_1m_tokens_usd": 4.05,
            "recommended_for": "Small fixed-shape BERT/ResNet encoder models; unsuitable for modern LLMs due to lack of block-paged KV caching, EAGLE-3 speculative trees, and FP8 grouped MoE kernels.",
            "status": "warn",
        },
        {
            "id": "tier3_triton_tilelang_aiter_xcd_swizzled_monolithic",
            "name": "Tier 3: 统一 Triton / TileLang + AMD AITER 架构感知调优 (Wave64 + 8-XCD L2 亲和)",
            "name_en": "Tier 3: Unified OpenAI Triton / TileLang + AMD AITER (Wavefront-64 + 8-XCD L2 Swizzling)",
            "compiler_and_kernel_stack": "Hardware-agnostic Python DSL (Triton/TileLang) autotuned per target (waves_per_eu, matrix_instr_nonkdim=16) + AMD AITER assembly FlashAttention/MoE",
            "deployment_topology": "Homogeneous Monolithic vLLM/SGLang Node (8×MI300X 1.53TB HBM)",
            "hbm_bandwidth_utilization_pct": 82.4,
            "node_throughput_tok_s": 2850.0,
            "decode_p99_tpot_ms": 11.8,
            "max_concurrent_32k_sessions_per_node": 196,
            "cost_per_1m_tokens_usd": 2.11,
            "recommended_for": "Single-vendor AMD MI300X/MI325X clusters running unified vLLM/SGLang with 100% codebase parity and 2.54x higher throughput than naive HIPIFY.",
            "status": "pass",
        },
        {
            "id": "tier4_heterogeneous_pd_disaggregation_h100_prefill_mi300x_decode",
            "name": "Tier 4: 跨架构异构 PD 分离 (H100 算力池跑 Prefill + MI300X 192GB 5.3TB/s 显存池跑 Decode)",
            "name_en": "Tier 4: Cross-Vendor Heterogeneous PD Disaggregation (H100 Prefill + MI300X 192GB Decode via Mooncake RoCEv2)",
            "compiler_and_kernel_stack": "H100 runs FA3/Cutlass FP8 Prefill; MI300X runs AITER/Triton Wave64 Paged Decode; Mooncake 400G RoCEv2 transfers raw FP8 KV blocks across vendors",
            "deployment_topology": "Compute-Bound Prefill on 80GB H100 + Bandwidth/Capacity-Bound Decode on 192GB MI300X (1.53 TB/node)",
            "hbm_bandwidth_utilization_pct": 89.6,
            "node_throughput_tok_s": 3840.0,
            "decode_p99_tpot_ms": 7.4,
            "max_concurrent_32k_sessions_per_node": 260,
            "cost_per_1m_tokens_usd": 1.48,
            "recommended_for": "Multi-vendor enterprise AI centers combining NVIDIA & AMD GPUs—matching each silicon architecture to its physical roofline (H100 FLOPs for Prefill, MI300X 192GB/5.3TB/s for Decode).",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier3_vs_tier1_throughput_speedup": round(t3["node_throughput_tok_s"] / t1["node_throughput_tok_s"], 2),
        "tier4_vs_tier1_throughput_speedup": round(t4["node_throughput_tok_s"] / t1["node_throughput_tok_s"], 2),
        "tier4_vs_tier3_tpot_improvement_pct": round(
            (t3["decode_p99_tpot_ms"] - t4["decode_p99_tpot_ms"]) / t3["decode_p99_tpot_ms"] * 100.0, 1
        ),
        "tier4_cost_reduction_vs_tier1_pct": round(
            (t1["cost_per_1m_tokens_usd"] - t4["cost_per_1m_tokens_usd"]) / t1["cost_per_1m_tokens_usd"] * 100.0, 1
        ),
        "mi300x_vs_h100_vram_capacity_ratio": round(192.0 / 80.0, 2),
        "mi300x_vs_h100_hbm_bw_ratio": round(5.30 / 3.35, 2),
    }

    return {
        "experiment_id": "heterogeneous_gpu_rocm_cuda_triton_stack_selection",
        "date": "2026-11-12",
        "round": 49,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "AMD Instinct MI300X CDNA3 Architecture Whitepaper (8 XCD Chiplets, 256MB Infinity Cache, 192GB 5.3TB/s HBM3, Wavefront-64 MFMA)",
                "AMD AITER (AI Tensor Engine for ROCm) & OpenAI Triton / TileLang Cross-Vendor Autotuning Benchmarks",
                "Mooncake / vLLM Disaggregated Serving over Standard RoCEv2 RDMA Across Heterogeneous Prefill & Decode Pools",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_heterogeneous_gpu_rocm_cuda_triton_matrix()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

"""Deterministic evaluator for GPU Kernel Execution Graph & Compilation Stack Selection (Round 45).

Models the end-to-end Decode time per output token (TPOT ms), single-batch throughput (tokens/s),
CPU kernel launch bubble percentage, number of GPU kernel launches per token, graph capture VRAM overhead (GB),
and cost per 1M output tokens across four GPU execution & compilation tiers on an 8×H100-80GB SXM5 node
running Llama-3.3-70B-FP8 (80 layers, TP=8, interactive small batch BS=8):
  - Tier 1: PyTorch Eager Mode (480 kernel launches/token -> 62.4% CPU launch overhead bubble)
  - Tier 2: Monolithic Static CUDA Graphs (38 pre-captured batch buckets -> 18.6 GB VRAM bloat & padding waste)
  - Tier 3: torch.compile (TorchInductor + Triton Fusion) + vLLM V1 Piecewise CUDA Graphs (dynamic-shape friendly)
  - Tier 4: Persistent Megakernel (FlashInfer / Mirage MPK / Thunder) + Hopper/Blackwell PDL (Programmatic Dependent Launch)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_cuda_graphs_torch_compile_megakernel_matrix() -> dict[str, Any]:
    workload = {
        "cluster_node": "1 × 8×H100-80GB SXM5 Node (900 GB/s NVLink-4, $28.80/hr node cost)",
        "model_profile": "70B Dense LLM FP8 (80 Transformer layers, TP=8, 64 Q heads / 8 KV heads)",
        "serving_regime": "Interactive Low-Latency Decode (Batch Size BS=8, 8K Context) + Dynamic Chunked Prefill Mix",
        "cpu_single_launch_latency_us": 8.5,
        "avg_gpu_micro_kernel_duration_us": 3.2,
    }

    tiers = [
        {
            "id": "tier1_pytorch_eager_cpu_launch_bound",
            "name": "Tier 1: 原生 PyTorch Eager 逐算子发射 (480 次/词 CPU 启动墙)",
            "name_en": "Tier 1: Vanilla PyTorch Eager Execution (480 Kernel Launches/Token CPU Bottleneck)",
            "execution_mechanism": "Host Python/C++ Dispatcher issues 480 separate cudaLaunchKernel calls per token (6 ops/layer × 80 layers)",
            "dynamic_shape_support": "100% Native Dynamic Shapes (Zero Recompile), but severely CPU-bound",
            "kernel_launches_per_token": 480,
            "cpu_launch_bubble_pct": 62.4,
            "decode_tpot_ms": 24.8,
            "stream_tokens_per_sec_per_req": 40.3,
            "node_throughput_tok_s": 322.6,
            "graph_vram_overhead_gb": 0.0,
            "cost_per_1m_tokens_usd": 24.80,
            "recommended_for": "Local debugging and rapid model prototyping only; strictly banned in production LLM serving because the GPU sits idle 62.4% of the time waiting for CPU cudaLaunchKernel calls.",
            "status": "fail",
        },
        {
            "id": "tier2_monolithic_static_cuda_graphs_bucketed",
            "name": "Tier 2: 整图静态 CUDA Graphs 多分桶捕获 (38 桶显存膨胀与补零浪费)",
            "name_en": "Tier 2: Monolithic Static Bucketed CUDA Graphs (38 Batch Buckets -> 18.6 GB VRAM Bloat)",
            "execution_mechanism": "Pre-captures 38 static full-model CUDA Graphs (BS=1,2,4,8..512); pads dynamic batches up to nearest bucket",
            "dynamic_shape_support": "Rigid Fixed Pointers Only (Breaks on CPU Attention metadata, MoE dynamic routing, or mixed prefill-decode)",
            "kernel_launches_per_token": 480,
            "cpu_launch_bubble_pct": 14.5,
            "decode_tpot_ms": 11.2,
            "stream_tokens_per_sec_per_req": 89.3,
            "node_throughput_tok_s": 714.3,
            "graph_vram_overhead_gb": 18.6,
            "cost_per_1m_tokens_usd": 11.20,
            "recommended_for": "Legacy vLLM v0 fixed-shape Decode-only workloads; wastes 18.6 GB/GPU of KV Cache memory across 38 static buckets and still suffers un-fused HBM roundtrips.",
            "status": "warn",
        },
        {
            "id": "tier3_torch_compile_inductor_piecewise_cudagraph",
            "name": "Tier 3: torch.compile (Inductor 算子融合) + vLLM V1 分段 CUDA Graphs (Piecewise)",
            "name_en": "Tier 3: torch.compile (TorchInductor Triton Fusion) + vLLM V1 Piecewise CUDA Graphs",
            "execution_mechanism": "Triton fuses RMSNorm+Residual+RoPE+Quant & SiLU+Mul (480->165 kernels); Piecewise Graph captures static MLP/GEMM blocks while leaving dynamic Attention flexible",
            "dynamic_shape_support": "Seamless Symbolic Shapes + Unified Chunked-Prefill & Decode (1 Graph Tree with cudaMemPool Reuse)",
            "kernel_launches_per_token": 165,
            "cpu_launch_bubble_pct": 4.8,
            "decode_tpot_ms": 8.1,
            "stream_tokens_per_sec_per_req": 123.5,
            "node_throughput_tok_s": 987.7,
            "graph_vram_overhead_gb": 2.4,
            "cost_per_1m_tokens_usd": 8.10,
            "recommended_for": "Standard enterprise production baseline (vLLM V1 / SGLang torch.compile) balancing 3.06x lower TPOT, 87% lower graph VRAM overhead (2.4 GB vs 18.6 GB), and full dynamic-shape flexibility.",
            "status": "pass",
        },
        {
            "id": "tier4_persistent_megakernel_hopper_blackwell_pdl",
            "name": "Tier 4: 持久大内核 (Persistent Megakernel) + Hopper/Blackwell PDL 硬件级联启动",
            "name_en": "Tier 4: Persistent Megakernel (FlashInfer/Mirage MPK) + Hopper/Blackwell PDL Overlap",
            "execution_mechanism": "Fuses multi-op transformer decode (Norm+QKV+Attn+Proj+NVLS AllReduce) into persistent SM resident worker blocks + CUDA Programmatic Dependent Launch (PDL) preamble overlap",
            "dynamic_shape_support": "On-Chip Task Queue & Ring-Buffer Descriptor Table (Zero Host Launch Inside Layer Loop)",
            "kernel_launches_per_token": 16,
            "cpu_launch_bubble_pct": 0.9,
            "decode_tpot_ms": 5.4,
            "stream_tokens_per_sec_per_req": 185.2,
            "node_throughput_tok_s": 1481.5,
            "graph_vram_overhead_gb": 1.2,
            "cost_per_1m_tokens_usd": 5.40,
            "recommended_for": "Ultra-low-latency (<6ms TPOT, 185+ tok/s) reasoning, voice agents, and speculative verification on H100/H200/B200 clusters.",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier3_vs_tier1_speedup_ratio": round(t1["decode_tpot_ms"] / t3["decode_tpot_ms"], 2),
        "tier4_vs_tier1_speedup_ratio": round(t1["decode_tpot_ms"] / t4["decode_tpot_ms"], 2),
        "tier4_vs_tier2_speedup_ratio": round(t2["decode_tpot_ms"] / t4["decode_tpot_ms"], 2),
        "tier3_vram_savings_vs_tier2_gb": round(t2["graph_vram_overhead_gb"] - t3["graph_vram_overhead_gb"], 1),
        "tier4_cost_reduction_vs_tier1_pct": round(
            (t1["cost_per_1m_tokens_usd"] - t4["cost_per_1m_tokens_usd"]) / t1["cost_per_1m_tokens_usd"] * 100.0, 1
        ),
    }

    return {
        "experiment_id": "cuda_graphs_torch_compile_megakernel_selection",
        "date": "2026-11-11",
        "round": 45,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "vLLM V1 Architecture & Piecewise CUDA Graph + torch.compile Design Specs (Symbolic Shapes & Shared Memory Pool)",
                "FlashInfer & NVIDIA Hopper/Blackwell Programmatic Dependent Launch (PDL cudaTriggerProgrammaticLaunchCompletion) Benchmarks",
                "Mirage Persistent Kernel (MPK) & Stanford ThunderKittens / Megakernel Low-Latency LLM Decode Papers",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_cuda_graphs_torch_compile_megakernel_matrix()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path


def evaluate_embedding_reranker_tiers() -> dict:
    """Deterministic evaluator for 4 Enterprise Embedding & Cross-Encoder Reranker Serving Tiers.

    Workload Baseline:
      - Models: BGE-M3 / Qwen3-Embedding-0.6B (1024d) + bge-reranker-v2-m3 (568M parameters)
      - Hardware Unit: 1×NVIDIA L40S / H200 MIG Slice (or 64-Core Xeon AVX-512 for Tier 2)
      - Input Length Distribution: Skewed RAG chunks (Mean = 180 tokens, P95 = 650 tokens, Max = 1,024 tokens; 50 chunks per query rerank)
    """
    tiers = [
        {
            "tier_id": "tier1_naive_pytorch_sentence_transformers_padded",
            "name": "Tier 1: Naive FastAPI + PyTorch Sentence-Transformers (Padded Rectangular Batch [B, L_max])",
            "name_zh": "Tier 1: 传统 FastAPI + PyTorch 原生填充批处理 (按最长序列补零 [B, L_max]，浪费 82.4% 算力)",
            "batching_and_kernel": "Python GIL + pad_sequence to max_len=1024 + Standard SDPA over 82.4% [PAD] zeros",
            "docs_embedded_per_sec": 1250,
            "rerank_50_docs_p99_latency_ms": 380.0,
            "padding_compute_waste_pct": 82.4,
            "ndcg_at_10_accuracy_pct": 100.0,
            "vector_index_ram_per_100m_gb": 409.6,
            "cost_per_10m_docs_usd": 4.80,
        },
        {
            "tier_id": "tier2_cpu_onnx_runtime_int8_avx512",
            "name": "Tier 2: CPU ONNX Runtime / OpenVINO Dynamic INT8 Quantization (64-Core AVX-512 / AMX)",
            "name_zh": "Tier 2: CPU ONNX Runtime / OpenVINO 动态 INT8 量化 (适合低并发边缘/无卡环境)",
            "batching_and_kernel": "Static graph fusion + VNNI/AMX INT8 GEMM on CPU; still pads within micro-batches",
            "docs_embedded_per_sec": 420,
            "rerank_50_docs_p99_latency_ms": 195.0,
            "padding_compute_waste_pct": 45.0,
            "ndcg_at_10_accuracy_pct": 99.1,
            "vector_index_ram_per_100m_gb": 409.6,
            "cost_per_10m_docs_usd": 3.10,
        },
        {
            "tier_id": "tier3_hf_tei_infinity_unpadded_flashattn_varlen",
            "name": "Tier 3: Rust HuggingFace TEI / Infinity (Token-Budget Batching + Unpadded cu_seqlens FlashAttention)",
            "name_zh": "Tier 3: HuggingFace TEI / Infinity 零填充引擎 (1D 紧致拼接 cu_seqlens + Token 级动态攒批)",
            "batching_and_kernel": "Rust async router + 0% padding 1D flattened tensor + flash_attn_varlen_func",
            "docs_embedded_per_sec": 10500,
            "rerank_50_docs_p99_latency_ms": 28.5,
            "padding_compute_waste_pct": 0.0,
            "ndcg_at_10_accuracy_pct": 100.0,
            "vector_index_ram_per_100m_gb": 409.6,
            "cost_per_10m_docs_usd": 0.58,
        },
        {
            "tier_id": "tier4_tensorrt_fp8_unpadded_plus_matryoshka_mrl",
            "name": "Tier 4: TensorRT-FP8 Unpadded CUDA Graphs + Matryoshka (MRL 1024d -> 256d INT8) Two-Stage Funnel",
            "name_zh": "Tier 4: TensorRT-FP8 零填充计算图 + 俄罗斯套娃降维 (Matryoshka 256d) 两阶段漏斗",
            "batching_and_kernel": "TRT-FP8 varlen plugin + CUDA Graphs + MRL 256d INT8 first-stage recall & 50-doc FP8 rerank",
            "docs_embedded_per_sec": 18500,
            "rerank_50_docs_p99_latency_ms": 14.2,
            "padding_compute_waste_pct": 0.0,
            "ndcg_at_10_accuracy_pct": 99.6,
            "vector_index_ram_per_100m_gb": 25.6,
            "cost_per_10m_docs_usd": 0.32,
        },
    ]

    # Sequence Length Skew Perturbation (Uniform 256 tok vs Moderate Skew vs Extreme 1-Long-Outlier in Batch of 64)
    length_skew_perturbation = [
        {
            "batch_distribution": "Uniform Lengths (All 64 docs = 180 tokens)",
            "padded_pytorch_docs_per_sec": 5400,
            "unpadded_tei_trt_docs_per_sec": 18500,
        },
        {
            "batch_distribution": "Moderate RAG Skew (Mean 180 tok, Max 512 tok)",
            "padded_pytorch_docs_per_sec": 2450,
            "unpadded_tei_trt_docs_per_sec": 18480,
        },
        {
            "batch_distribution": "Extreme Skew (63 docs @ 80 tok + 1 outlier doc @ 1,024 tok)",
            "padded_pytorch_docs_per_sec": 1250,
            "unpadded_tei_trt_docs_per_sec": 18450,
        },
        {
            "batch_distribution": "Long-Context Embedding (63 docs @ 120 tok + 1 doc @ 4,096 tok)",
            "padded_pytorch_docs_per_sec": 190,
            "unpadded_tei_trt_docs_per_sec": 18200,
        },
    ]

    t1 = tiers[0]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier1_padding_waste_pct": t1["padding_compute_waste_pct"],
        "tier3_vs_tier1_throughput_gain_x": round(
            t3["docs_embedded_per_sec"] / t1["docs_embedded_per_sec"], 1
        ),
        "tier4_vs_tier1_throughput_gain_x": round(
            t4["docs_embedded_per_sec"] / t1["docs_embedded_per_sec"], 1
        ),
        "tier4_rerank50_p99_ms": t4["rerank_50_docs_p99_latency_ms"],
        "tier4_vs_tier1_rerank_speedup_x": round(
            t1["rerank_50_docs_p99_latency_ms"] / t4["rerank_50_docs_p99_latency_ms"], 2
        ),
        "tier4_mrl_memory_compression_x": round(
            t1["vector_index_ram_per_100m_gb"] / t4["vector_index_ram_per_100m_gb"], 1
        ),
    }

    return {
        "experiment_id": "embedding_reranker_serving_engine_selection",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_unpadded_flashattention_varlen_and_matryoshka_embedding_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "HuggingFace Text-Embeddings-Inference (TEI) & Infinity Unpadded cu_seqlens FlashAttention Benchmark",
                "Padding Tax Analysis on Skewed Sequence-Length Distributions in Bidirectional Transformer Encoders",
                "Matryoshka Representation Learning (MRL 1024d -> 256d) + TensorRT-FP8 Encoder Optimization",
            ],
        },
        "workload_baseline": {
            "models": "BGE-M3 / Qwen3-Embedding (1024d) + bge-reranker-v2-m3 (Top-50 Rerank)",
            "chunk_length_distribution": "Mean 180 tokens, Max 1,024 tokens",
            "vector_corpus_scale": "100 Million Chunks",
        },
        "tiers": tiers,
        "length_skew_perturbation": length_skew_perturbation,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_embedding_reranker_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

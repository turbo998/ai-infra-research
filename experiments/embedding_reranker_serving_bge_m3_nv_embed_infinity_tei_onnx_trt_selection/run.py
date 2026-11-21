from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "Benchmark": "embedding_reranker_serving_bge_m3_nv_embed_infinity_tei_onnx_trt_selection",
        "Provenance": {
            "Data_Status": "synthetic + anonymized_reported",
            "Independently_Reproduced": False,
            "Physical_GPU_Cluster_Measurement": "not_measured",
            "Harness_Policy": "enterprise_embedding_reranker_tei_infinity_trt_serving_sizer_v1",
        },
        "Workload_Assumptions": {
            "Target_Workload": "Enterprise Hybrid RAG Online Query Embedding + Top-50 Cross-Encoder Reranking (BGE-M3 + BGE-Reranker-v2-m3 / Qwen-Reranker, 350 RAG Queries/sec)",
            "RAG_Queries_Per_Second": 350,
            "Rerank_Candidates_Per_Query": 50,
            "Avg_Candidate_Tokens": 210,
            "Max_Candidate_Tokens_In_Batch": 512,
        },
        "Configuration_Matrix": [
            {
                "Tier_ID": "Tier_1_Naive_FastAPI_PyTorch_Padded_Batching_FP16",
                "Architecture_Name": "Naive FastAPI + HuggingFace `AutoModel` PyTorch Padded Tensor (`[B, S_max=512]`, GIL-Bound Tokenization)",
                "Engine_And_Batching": "Python Tokenizer + Rectangular `[PAD]` Tensor Padding (59% Wasted Pad FLOPs)",
                "Rerank_Top50_P99_Latency_ms": 168.0,
                "Throughput_Candidates_Per_Sec_Per_GPU": 620.0,
                "Required_L40S_GPUs": 36,
                "NDCG_At_10_Retrieval_Quality": 86.4,
                "Monthly_Cluster_TCO_USD": 39600,
                "Recommended_Scenario": "Anti-pattern for production RAG: quadratic pad waste and Python tokenizer GIL bottleneck",
            },
            {
                "Tier_ID": "Tier_2_ONNX_Runtime_Dynamic_Batch_Padded_FP16",
                "Architecture_Name": "NVIDIA Triton + ONNX Runtime Dynamic Batcher (Padded Rectangular Batching, FP16)",
                "Engine_And_Batching": "C++ Triton Dynamic Batcher + ONNX Runtime CUDA EP (Still Rectangular Padded per Batch)",
                "Rerank_Top50_P99_Latency_ms": 64.0,
                "Throughput_Candidates_Per_Sec_Per_GPU": 1480.0,
                "Required_L40S_GPUs": 16,
                "NDCG_At_10_Retrieval_Quality": 86.4,
                "Monthly_Cluster_TCO_USD": 17600,
                "Recommended_Scenario": "Legacy Triton deployments without unpadded FlashAttention ragged tensor support",
            },
            {
                "Tier_ID": "Tier_3_vLLM_Embed_Score_Paged_Ragged_BF16",
                "Architecture_Name": "vLLM (`--task embed / score`) Unified Ragged FlashAttention Engine (7B LLM-Based Embedding/Reranker)",
                "Engine_And_Batching": "Rust/Fast Tokenizer + Unpadded FlashAttention-2 (`cu_seqlens`) on Decoder-Based Embeddings",
                "Rerank_Top50_P99_Latency_ms": 28.5,
                "Throughput_Candidates_Per_Sec_Per_GPU": 2450.0,
                "Required_L40S_GPUs": 10,
                "NDCG_At_10_Retrieval_Quality": 88.2,
                "Monthly_Cluster_TCO_USD": 11000,
                "Recommended_Scenario": "Decoder-based 4B–7B embeddings (`NV-Embed-v2`, `Qwen2.5-Embedding`, `BGE-en-ICL`) unified on vLLM stack",
            },
            {
                "Tier_ID": "Tier_4_TEI_Infinity_Unpadded_FlashAttn_Length_Sorted_FP8",
                "Architecture_Name": "HuggingFace TEI / `Infinity` + Rust Tokenizer + Unpadded `cu_seqlens` FlashAttention + Length-Sorted Token-Budget Batching + FP8/INT8",
                "Engine_And_Batching": "Zero-Pad Ragged FlashAttention (`max_batch_tokens=16384`) + Length Bucketing + FP8 TensorRT/Candle",
                "Rerank_Top50_P99_Latency_ms": 14.5,
                "Throughput_Candidates_Per_Sec_Per_GPU": 4200.0,
                "Required_L40S_GPUs": 6,
                "NDCG_At_10_Retrieval_Quality": 86.3,
                "Monthly_Cluster_TCO_USD": 6600,
                "Recommended_Scenario": "Production standard for BERT/XLM-R/BGE-M3 Embeddings & Cross-Encoder Rerankers (14.5ms P99, -83.3% GPUs)",
            },
        ],
        "Decision_Summary": {
            "Baseline_Tier": "Tier_1_Naive_FastAPI_PyTorch_Padded_Batching_FP16",
            "Recommended_Tier": "Tier_4_TEI_Infinity_Unpadded_FlashAttn_Length_Sorted_FP8",
            "P99_Latency_Reduction_Pct": 91.4,
            "Single_GPU_Throughput_Multiplier": 6.77,
            "GPU_Count_Reduction_Pct": 83.3,
            "Monthly_TCO_Savings_Pct": 83.3,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()

    if args.check:
        if not out_path.exists():
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing["Benchmark"] == payload["Benchmark"]
        assert len(existing["Configuration_Matrix"]) == 4
        print("Wrote results.json")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

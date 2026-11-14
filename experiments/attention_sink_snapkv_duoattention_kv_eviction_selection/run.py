"""Deterministic evaluator for Long-Context KV Cache Budget Control, Attention Sinks & Retrieval-Head Separation (StreamingLLM vs. H2O vs. SnapKV vs. DuoAttention) Selection (Round 184).

Models KV cache memory per 128K-token request on a 70B model (GB),
128K Needle-in-a-Haystack (NIAH) retrieval accuracy (%), streaming perplexity stability,
FlashAttention-3 / PagedAttention compatibility, and Decode TPOT (ms) across four KV budget tiers:
  - Tier 1: Naive Sliding Window Without Attention Sinks (Evicts Token 0..3 -> Softmax Collapse, Perplexity >1,200)
  - Tier 2: StreamingLLM (4 Attention Sink Tokens + 4K Sliding Window -> Infinite Fluency, but 14.5% Mid-Context NIAH Recall)
  - Tier 3: Per-Step Dynamic Heavy-Hitter Eviction (`H2O`: Needs Explicit Attention Score Matrix -> Incompatible with FlashAttention)
  - Tier 4: DuoAttention + SnapKV (Full KV on 20% Retrieval Heads + Sink/Window on 80% Streaming Heads = 99.2% NIAH Recall, 3.5x Smaller KV)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_attention_sink_snapkv_duoattention_kv_eviction_matrix() -> dict[str, Any]:
    workload = {
        "model_and_context": "70B GQA Model (80 Layers, 8 KV Heads/Layer) at 128,000-Token Context (Full FP16 KV Cache = 40.0 GB per request!)",
        "attention_sink_physics": "Because Softmax(QK^T) must sum to 1.0 even when the current token has no relevant match in history, autoregressive LLMs dump 45–80% of unneeded attention probability onto the very first 1–4 tokens (Attention Sinks). Evicting Token 0 breaks the Softmax denominator!",
        "head_specialization": "Only ~20% of KV heads are 'Retrieval Heads' (attending globally across 128K tokens); the other ~80% are 'Streaming Heads' (only ever attending to the 4 Sink tokens + local recent window)!",
    }

    tiers = [
        {
            "id": "tier1_naive_sliding_window_no_sinks",
            "name": "Tier 1: 朴素滑动窗口 (直接丢弃开头第 0~3 个 Token · 触发 Softmax 崩溃乱码 ✕)",
            "name_en": "Tier 1: Naive Sliding Window Without Attention Sinks (Evicts Initial Tokens -> Softmax Perplexity Collapse)",
            "kv_memory_128k_gb": 2.50,
            "niah_128k_recall_pct": 8.0,
            "streaming_perplexity": 1420.0,
            "decode_tpot_ms": 6.4,
            "flashattention_compatible": True,
            "recommended_for": "Never use on pre-trained autoregressive LLMs without retraining; evicting Token 0..3 destroys the Softmax normalization sink.",
            "status": "fail",
        },
        {
            "id": "tier2_streamingllm_4sinks_plus_local_window",
            "name": "Tier 2: StreamingLLM (保留开头 4 个 Sink Token + 局部滑动窗口 · 无限流式不崩但丢中间事实)",
            "name_en": "Tier 2: StreamingLLM (4 Attention Sinks + Local Sliding Window -> Stable Perplexity, Loses Mid-Document Facts)",
            "kv_memory_128k_gb": 2.52,
            "niah_128k_recall_pct": 14.5,
            "streaming_perplexity": 5.3,
            "decode_tpot_ms": 6.5,
            "flashattention_compatible": True,
            "recommended_for": "Infinite-turn casual voice/companion chat or live stream captioning where only the last few turns matter (not for long-doc RAG).",
            "status": "warn",
        },
        {
            "id": "tier3_h2o_dynamic_heavy_hitter_eviction",
            "name": "Tier 3: H2O 逐跳动态累计注意力驱逐 (需显式算注意力权重矩阵 · 与 FlashAttention 冲突)",
            "name_en": "Tier 3: H2O Dynamic Heavy-Hitter Oracle (Requires Explicit Attention Matrix, Breaks FlashAttention Fusion)",
            "kv_memory_128k_gb": 10.00,
            "niah_128k_recall_pct": 84.5,
            "streaming_perplexity": 5.2,
            "decode_tpot_ms": 24.8,
            "flashattention_compatible": False,
            "recommended_for": "Research baselines; maintaining per-step cumulative attention scores requires materializing O(N) scores and fragments PagedAttention.",
            "status": "warn",
        },
        {
            "id": "tier4_duoattention_snapkv_retrieval_head_split",
            "name": "Tier 4: DuoAttention (20% 检索头保留全量 KV + 80% 流式头仅存 Sink+滑窗) + SnapKV 前缀压缩 🏆",
            "name_en": "Tier 4: DuoAttention (20% Retrieval Heads Full KV + 80% Streaming Heads Sink+Window) + SnapKV Observation Pooling",
            "kv_memory_128k_gb": 11.40,
            "niah_128k_recall_pct": 99.2,
            "streaming_perplexity": 5.1,
            "decode_tpot_ms": 8.2,
            "flashattention_compatible": True,
            "recommended_for": "Production 128K–1M long-context RAG & Coding Agents: shrinks KV cache by 71.5% (40GB -> 11.4GB) while retaining 99.2% NIAH recall and full FlashAttention speed.",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier4_kv_memory_reduction_vs_full_40gb_pct": round(
            (40.0 - t4["kv_memory_128k_gb"]) / 40.0 * 100.0, 1
        ),
        "tier4_vs_tier2_niah_recall_gain_pts": round(
            t4["niah_128k_recall_pct"] - t2["niah_128k_recall_pct"], 1
        ),
        "tier4_vs_tier3_decode_speedup_ratio": round(
            t3["decode_tpot_ms"] / t4["decode_tpot_ms"], 2
        ),
        "tier4_niah_recall_pct": t4["niah_128k_recall_pct"],
    }

    return {
        "experiment_id": "attention_sink_snapkv_duoattention_kv_eviction_selection",
        "date": "2026-11-15",
        "round": 184,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_published_kv_compression_papers",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "StreamingLLM: Efficient Streaming Language Models with Attention Sinks (MIT HAN Lab / ICLR)",
                "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads (MIT HAN Lab)",
                "SnapKV: LLM Knows What You are Looking for Before Generation & H2O (Heavy-Hitter Oracle) Benchmarks",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_attention_sink_snapkv_duoattention_kv_eviction_matrix()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

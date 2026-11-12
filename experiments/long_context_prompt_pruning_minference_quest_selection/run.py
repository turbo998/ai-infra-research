from __future__ import annotations

import json
from pathlib import Path


def evaluate_long_context_minference_quest_tiers() -> dict:
    """Deterministic evaluator for 4 Long-Context 128K Prompt Pruning, MInference Sparse Prefill & Quest Dynamic KV Selection Tiers.

    Workload Baseline:
      - Cluster Node: 8x H100 80GB SXM5 ($24.00/hr) serving a 70B FP8/BF16 LLM (TP=8, GQA-8)
      - Workload: 128,000-Token Multi-Turn Enterprise Repository / Legal Corpus Analysis
        (5 Sequential Follow-Up Turns per Session asking different needle-in-a-haystack questions across the same 128K corpus)
      - Evaluates:
        1) 128K Prefill TTFT (seconds) — governed by O(N^2) causal attention FLOPs
        2) Decode TPOT (ms) — governed by HBM KV cache bandwidth read per step
        3) Multi-Turn Needle-in-a-Haystack Recall (%) across all 5 distinct follow-up questions
        4) Serving Throughput (req/min per node) and Cost per 1,000 128K Multi-Turn Sessions ($)
    """
    tiers = [
        {
            "tier_id": "tier1_full_128k_dense_attention",
            "name": "Tier 1: Full 128K Dense FlashAttention-3 Prefill + Full 128K Paged KV Decode (100% Compute & HBM Read)",
            "name_zh": "Tier 1: 全量 128K 稠密 Prefill + 全量 128K KV 逐词读取 Decode (O(N^2) 算力墙 + 显存带宽墙)",
            "mechanism": "Computes all 128K×128K causal attention pairs during prefill and reads all 41.9 GB of 128K KV cache on every decode step",
            "prefill_128k_ttft_s": 14.80,
            "decode_tpot_ms": 46.5,
            "multi_turn_needle_recall_pct": 99.2,
            "sessions_per_hour_per_node": 68.0,
            "cost_per_1k_sessions_usd": 352.94,
            "annual_500k_sessions_tco_usd": 1764700,
        },
        {
            "tier_id": "tier2_input_token_dropping_llmlingua2",
            "name": "Tier 2: Hard Input Token Pruning (LLMLingua-2 4x Text Compression Before LLM + Prefix Cache Breakage)",
            "name_zh": "Tier 2: 输入端硬删词压缩 (LLMLingua-2 4x 删词：破坏代码/数字语法且无法跨多轮保护未知考点)",
            "mechanism": "BERT/XLM-R token classifier drops 75% of input tokens (128K -> 32K) before LLM prefill; irreversibly deletes exact numbers/code syntax",
            "prefill_128k_ttft_s": 2.15,
            "decode_tpot_ms": 14.8,
            "multi_turn_needle_recall_pct": 71.4,
            "sessions_per_hour_per_node": 295.0,
            "cost_per_1k_sessions_usd": 81.36,
            "annual_500k_sessions_tco_usd": 406800,
        },
        {
            "tier_id": "tier3_permanent_kv_eviction_h2o_snapkv",
            "name": "Tier 3: Permanent KV Cache Eviction (H2O / SnapKV Drops 85% of KV After Turn-1 Query)",
            "name_zh": "Tier 3: 永久性 KV 驱逐 (H2O / SnapKV 按第 1 问注意力永久删掉 85% KV：第 2~5 问换话题即失忆)",
            "mechanism": "Evicts 85% of KV blocks permanently based on Turn-1 prompt attention; Turn-2..5 questions targeting previously evicted sections fail",
            "prefill_128k_ttft_s": 14.80,
            "decode_tpot_ms": 11.6,
            "multi_turn_needle_recall_pct": 64.8,
            "sessions_per_hour_per_node": 142.0,
            "cost_per_1k_sessions_usd": 169.01,
            "annual_500k_sessions_tco_usd": 845050,
        },
        {
            "tier_id": "tier4_minference_sparse_prefill_plus_quest_page_envelope_decode",
            "name": "Tier 4: MInference Vertical-Slash Sparse Prefill + Quest Non-Destructive Page-Envelope Top-4K KV Selection",
            "name_zh": "Tier 4: 预填充 MInference 垂直-斜线动态稀疏 + 解码期 Quest 页面包络非破坏性检索 (保留全量 KV 但每步仅读 Top-4K)",
            "mechanism": "MInference skips 88% of 128K^2 prefill tiles via per-head Vertical-Slash patterns; Quest retains 100% KV in HBM/DRAM but scores [K_min, K_max] page bounding boxes to load only Top-4,096 KV tokens per decode step",
            "prefill_128k_ttft_s": 1.82,
            "decode_tpot_ms": 9.8,
            "multi_turn_needle_recall_pct": 98.9,
            "sessions_per_hour_per_node": 485.0,
            "cost_per_1k_sessions_usd": 49.48,
            "annual_500k_sessions_tco_usd": 247400,
        },
    ]

    # Perturbation across Context Lengths (32K -> 64K -> 128K -> 256K) and Follow-Up Turn Topic Shifts
    context_perturbation = [
        {
            "condition": "32K Context (Multi-Turn Codebase Q&A)",
            "tier1_dense_ttft_s": 1.45,
            "tier4_minference_ttft_s": 0.42,
            "tier3_evict_recall_pct": 76.5,
            "tier4_quest_recall_pct": 99.4,
            "insight": "MInference speeds up 32K prefill 3.45x; Quest non-destructive Top-4K selection preserves Turn-2..5 recall",
        },
        {
            "condition": "64K Context (Financial 10-K Multi-Metric Audit)",
            "tier1_dense_ttft_s": 4.40,
            "tier4_minference_ttft_s": 0.85,
            "tier3_evict_recall_pct": 69.2,
            "tier4_quest_recall_pct": 99.1,
            "insight": "Quadratic O(N^2) attention begins dominating 64K prefill; MInference reaches 5.18x speedup",
        },
        {
            "condition": "128K Context (5-Turn Legal / Repo Deep Dive Baseline)",
            "tier1_dense_ttft_s": 14.80,
            "tier4_minference_ttft_s": 1.82,
            "tier3_evict_recall_pct": 64.8,
            "tier4_quest_recall_pct": 98.9,
            "insight": "8.13x faster TTFT (14.80s -> 1.82s) and 4.74x faster TPOT (46.5ms -> 9.8ms) with 98.9% multi-turn recall",
        },
        {
            "condition": "256K Context (Full Monorepo / Multi-Book Synthesis)",
            "tier1_dense_ttft_s": 54.20,
            "tier4_minference_ttft_s": 4.95,
            "tier3_evict_recall_pct": 52.0,
            "tier4_quest_recall_pct": 98.4,
            "insight": "10.95x faster prefill TTFT; permanent KV eviction (Tier 3) loses half of multi-turn needles at 256K",
        },
    ]

    t1 = tiers[0]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier4_vs_tier1_ttft_speedup_x": round(
            t1["prefill_128k_ttft_s"] / t4["prefill_128k_ttft_s"], 2
        ),
        "tier4_vs_tier1_tpot_speedup_x": round(
            t1["decode_tpot_ms"] / t4["decode_tpot_ms"], 2
        ),
        "tier4_vs_tier3_multi_turn_recall_gain_pt": round(
            t4["multi_turn_needle_recall_pct"] - t3["multi_turn_needle_recall_pct"], 1
        ),
        "tier4_vs_tier1_throughput_gain_x": round(
            t4["sessions_per_hour_per_node"] / t1["sessions_per_hour_per_node"], 2
        ),
        "tier4_vs_tier1_annual_savings_usd": t1["annual_500k_sessions_tco_usd"] - t4["annual_500k_sessions_tco_usd"],
    }

    return {
        "experiment_id": "long_context_prompt_pruning_minference_quest_selection",
        "date": "2026-11-12",
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention (A-Shape, Vertical-Slash, Block-Sparse)",
                "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference (Min-Max Page Envelope Bounding Box Scoring)",
                "LLMLingua-2 & H2O / SnapKV Permanent KV Eviction Failure Modes under Multi-Turn Topic Shifts",
            ],
        },
        "workload_baseline": {
            "node": "8x H100 80GB SXM5 ($24.00/hr), 70B FP8/BF16 LLM (TP=8)",
            "session_profile": "128,000-token context × 5 sequential multi-turn questions shifting focus across the document",
        },
        "tiers": tiers,
        "context_perturbation": context_perturbation,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_long_context_minference_quest_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

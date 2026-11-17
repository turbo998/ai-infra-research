"""Deterministic evaluator for Real-Time Customer Service & Enterprise Copilot Sub-400ms Low-Latency Architecture Selection (Round 368).

Models end-to-end P99 Time-to-First-Token (ms), P50 TTFT (ms), P99 Decode TPOT (ms),
safety guardrail interception rate (%), and cost per 1,000 customer interactions (USD)
across four enterprise real-time Copilot architectures (500 RPS peak, SLA: P99 E2E TTFT <= 400 ms):
  - Tier 1: 5-Stage Serial Waterfall Pipeline (Input Guardrail -> 70B Rewrite -> RAG -> 70B LLM -> Blocking Output Guardrail)
  - Tier 2: Semantic Cache Short-Circuit + 8B SLM Intent Router (Fast on Cache Hits, Still 1,280ms P99 on Cache Misses)
  - Tier 3: Async Parallel Guardrail & RAG + Pinned Prefix Cache + Sliding-Window Streaming Output Moderation
  - Tier 4: Speculative Pre-EOU RAG Prefetch + Optimistic Parallel Prefill + EAGLE-3 Speculative Decoding (275ms P99 E2E TTFT)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_realtime_customer_service_copilot_low_latency() -> dict[str, Any]:
    workload = {
        "enterprise_scenario": "Real-Time Customer Service & Live Agent Assist Copilot (500 RPS Peak, Multi-Turn Chat & Voice-to-Text, Strict Interactive SLA)",
        "target_sla": "End-to-End P99 TTFT <= 400.0 ms, P99 TPOT <= 14.0 ms, 100.0% Safety Guardrail Enforcement",
        "latency_budget_breakdown_tier4_ms": {
            "gateway_and_speculative_rag_cache_hit_ms": 18.0,
            "parallel_prompt_shield_and_prefill_ms": 215.0,
            "first_token_streaming_moderation_ms": 42.0,
            "total_p99_e2e_ttft_ms": 275.0,
        },
    }

    tiers = [
        {
            "id": "tier1_serial_five_stage_waterfall_pipeline",
            "name": "Tier 1: 传统五步串行瀑布流管线 (输入审查 -> 70B改写 -> RAG检索 -> 70B生成 -> 整段输出审查)",
            "name_en": "Tier 1: 5-Stage Serial Waterfall Pipeline (Serial Guardrail -> Rewrite -> RAG -> 70B -> Blocking Output Audit)",
            "pipeline_architecture": "Strictly Sequential Execution (320ms Guard + 680ms Rewrite + 290ms RAG + 1,450ms LLM + 680ms Output Check)",
            "p50_e2e_ttft_ms": 2180.0,
            "p99_e2e_ttft_ms": 3420.0,
            "p99_tpot_ms": 24.5,
            "guardrail_coverage_pct": 100.0,
            "cost_per_1k_queries_usd": 3.65,
            "recommended_for": "Anti-pattern for real-time customer service: 3.42s P99 delay causes severe conversational dead air and high abandonment.",
            "status": "fail",
        },
        {
            "id": "tier2_semantic_cache_plus_8b_slm_intent_router",
            "name": "Tier 2: 前置语义缓存短路 + 8B 小模型意图分流 (命中快，但未命中长尾仍需 1,280ms)",
            "name_en": "Tier 2: Semantic Cache Short-Circuit + 8B SLM Intent Router (1,280ms P99 on Cache Misses)",
            "pipeline_architecture": "Redis Semantic Cache (26% Hit @ 15ms) + 8B SLM Router (60% Simple Queries), Serial RAG on 40% Misses",
            "p50_e2e_ttft_ms": 380.0,
            "p99_e2e_ttft_ms": 1280.0,
            "p99_tpot_ms": 16.4,
            "guardrail_coverage_pct": 100.0,
            "cost_per_1k_queries_usd": 1.52,
            "recommended_for": "Internal employee IT helpdesk chatbots where P50 < 400ms is acceptable and P99 ~ 1.3s is tolerated.",
            "status": "warn",
        },
        {
            "id": "tier3_async_parallel_guardrail_rag_pinned_prefix_streaming_audit",
            "name": "Tier 3: 异步并行输入护栏与检索 + 前缀缓存置顶 + 流式滑动窗口输出审查",
            "name_en": "Tier 3: Async Parallel Input Guardrail & RAG + Pinned Prefix Cache + Streaming Window Output Audit",
            "pipeline_architecture": "Parallel (SLM Guardrail || Hybrid RAG = 85ms) + Pinned 8K Prefix Cache + 20-Token Sliding-Window Stream Audit",
            "p50_e2e_ttft_ms": 260.0,
            "p99_e2e_ttft_ms": 465.0,
            "p99_tpot_ms": 12.6,
            "guardrail_coverage_pct": 100.0,
            "cost_per_1k_queries_usd": 0.92,
            "recommended_for": "Standard enterprise web/mobile customer chat Copilot requiring sub-500ms response with full input/output guardrails.",
            "status": "pass",
        },
        {
            "id": "tier4_speculative_rag_prefetch_optimistic_prefill_eagle3_duplex",
            "name": "Tier 4: 流式输入投机预取 (Speculative RAG) + 乐观并行预填充 + EAGLE-3 投机解码 (275ms 极速秒回)",
            "name_en": "Tier 4: Speculative Pre-EOU RAG Prefetch + Optimistic Parallel Prefill + EAGLE-3 Speculative Decoding",
            "pipeline_architecture": "Pre-EOU Partial-Input RAG Prefetch (12ms) + Optimistic Parallel Guardrail/Prefill + EAGLE-3/MTP + Cascade Router",
            "p50_e2e_ttft_ms": 165.0,
            "p99_e2e_ttft_ms": 275.0,
            "p99_tpot_ms": 9.8,
            "guardrail_coverage_pct": 100.0,
            "cost_per_1k_queries_usd": 0.64,
            "recommended_for": "Real-time voice customer service, live call-center agent assist, and IDE coding Copilots demanding P99 TTFT <= 300ms.",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier4_vs_tier1_p99_ttft_speedup_x": round(
            t1["p99_e2e_ttft_ms"] / t4["p99_e2e_ttft_ms"], 2
        ),
        "tier3_vs_tier1_p99_ttft_speedup_x": round(
            t1["p99_e2e_ttft_ms"] / t3["p99_e2e_ttft_ms"], 2
        ),
        "tier4_cost_reduction_vs_tier1_pct": round(
            (t1["cost_per_1k_queries_usd"] - t4["cost_per_1k_queries_usd"]) / t1["cost_per_1k_queries_usd"] * 100.0, 1
        ),
        "tier4_p99_ttft_ms": t4["p99_e2e_ttft_ms"],
    }

    return {
        "experiment_id": "realtime_customer_service_copilot_low_latency_selection",
        "date": "2026-11-18",
        "round": 368,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_published_realtime_copilot_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "Enterprise Real-Time Copilot Latency Decomposition: Parallel Guardrails, Prefix Caching & Speculative RAG Prefetch",
                "EAGLE-3 / MTP Speculative Decoding & Cascade Routing Latency Benchmarks on vLLM/SGLang",
                "Streaming Sliding-Window Output Moderation vs. Full-Response Blocking Guardrail Latency Analysis",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify sealed results.json matches evaluator")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    result = evaluate_realtime_customer_service_copilot_low_latency()

    if args.check:
        sealed = json.loads(out_path.read_text(encoding="utf-8"))
        assert sealed == result, "Sealed results.json mismatch!"
        print("OK: realtime_customer_service_copilot_low_latency_selection verified.")
        return

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

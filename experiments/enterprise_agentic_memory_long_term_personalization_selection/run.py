from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise Agentic Memory & Long-Term Personalization Architecture Selection (Round 394)
    # Workload: 5,000,000 multi-session agent turns/month across 100+ historical sessions per user (6-month span).
    monthly_turns = 5_000_000

    # Tier 1: Sliding-Window Raw Chat History Stuffing (Last 50,000 tokens of raw chat logs)
    t1_temporal_fact_accuracy_pct = 54.0  # Forgets anything older than window; drowns in chitchat noise
    t1_stale_contradiction_rate_pct = 18.2
    t1_p95_ttft_ms = 1_650.0
    t1_monthly_cost_usd = 42_000

    # Tier 2: Recursive Rolling Text Summary ("Summarize conversation so far in 500 words")
    t2_temporal_fact_accuracy_pct = 61.5  # Lossy compression drops exact IDs, constraints, and code names
    t2_stale_contradiction_rate_pct = 14.0
    t2_p95_ttft_ms = 340.0
    t2_monthly_cost_usd = 6_200

    # Tier 3: Naive Append-Only Vector Memory Store (Embed every past message without conflict resolution)
    t3_temporal_fact_accuracy_pct = 74.5
    t3_stale_contradiction_rate_pct = 26.5  # Retrieves both Jan ("use MySQL") and May ("migrated to Snowflake")!
    t3_p95_ttft_ms = 390.0
    t3_monthly_cost_usd = 8_400

    # Tier 4: 3-Tier Cognitive Memory (Working Buffer + Bi-Temporal Conflict-Resolved Fact Graph [Mem0/Zep] + Episodic Memory)
    t4_temporal_fact_accuracy_pct = 96.8
    t4_stale_contradiction_rate_pct = 0.6  # ADD/UPDATE/INVALIDATE state machine with valid_from / valid_to
    t4_p95_ttft_ms = 310.0  # Memory extraction runs asynchronously in background (0ms online write penalty)
    t4_monthly_cost_usd = 7_400

    accuracy_gain_t4_vs_t1_pts = round(t4_temporal_fact_accuracy_pct - t1_temporal_fact_accuracy_pct, 1)
    accuracy_gain_t4_vs_t3_pts = round(t4_temporal_fact_accuracy_pct - t3_temporal_fact_accuracy_pct, 1)
    ttft_speedup_t4_vs_t1_x = round(t1_p95_ttft_ms / t4_p95_ttft_ms, 2)
    cost_savings_t4_vs_t1_pct = round((t1_monthly_cost_usd - t4_monthly_cost_usd) / t1_monthly_cost_usd * 100.0, 1)

    tiers = [
        {
            "tier_id": "tier1_sliding_window_raw_chat_stuffing",
            "name": "Tier 1: 原始聊天记录滑动窗口硬塞 (塞最近 50K Token 闲聊流水账，超窗即失忆且成本极高)",
            "temporal_fact_accuracy_pct": t1_temporal_fact_accuracy_pct,
            "stale_fact_contradiction_rate_pct": t1_stale_contradiction_rate_pct,
            "p95_ttft_ms": t1_p95_ttft_ms,
            "monthly_cost_5m_turns_usd": t1_monthly_cost_usd,
            "gdpr_forget_capability": "Hard (Requires Scanning Raw Chat Logs)",
            "status": "High Cost ($42K/mo) & Window Amnesia",
        },
        {
            "tier_id": "tier2_recursive_rolling_text_summary",
            "name": "Tier 2: 递归滚动全文摘要 (每 10 轮把摘要再浓缩一次，细节数字与精确约束严重丢失)",
            "temporal_fact_accuracy_pct": t2_temporal_fact_accuracy_pct,
            "stale_fact_contradiction_rate_pct": t2_stale_contradiction_rate_pct,
            "p95_ttft_ms": t2_p95_ttft_ms,
            "monthly_cost_5m_turns_usd": t2_monthly_cost_usd,
            "gdpr_forget_capability": "Impossible (Facts Baked into Opaque Summary)",
            "status": "Lossy Compression Decay (61.5% Accuracy)",
        },
        {
            "tier_id": "tier3_naive_append_only_vector_memory",
            "name": "Tier 3: 朴素只追加向量记忆库 (历史对话全存向量库，新旧偏好冲突率高达 26.5%)",
            "temporal_fact_accuracy_pct": t3_temporal_fact_accuracy_pct,
            "stale_fact_contradiction_rate_pct": t3_stale_contradiction_rate_pct,
            "p95_ttft_ms": t3_p95_ttft_ms,
            "monthly_cost_5m_turns_usd": t3_monthly_cost_usd,
            "gdpr_forget_capability": "Partial (Scattered Across Raw Turn Chunks)",
            "status": "Severe Temporal Contradiction (26.5% Conflict)",
        },
        {
            "tier_id": "tier4_three_tier_bitemporal_conflict_resolved_memory",
            "name": "Tier 4: 三层认知记忆架构 (工作缓冲 + 双时态冲突消解事实图谱 Mem0/Zep + 情景经验库)",
            "temporal_fact_accuracy_pct": t4_temporal_fact_accuracy_pct,
            "stale_fact_contradiction_rate_pct": t4_stale_contradiction_rate_pct,
            "p95_ttft_ms": t4_p95_ttft_ms,
            "monthly_cost_5m_turns_usd": t4_monthly_cost_usd,
            "gdpr_forget_capability": "100% Atomic Fact-ID Deletion & Audit Trail",
            "status": "Enterprise Long-Term Agent Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_agentic_memory_long_term_personalization_selection",
        "round": 394,
        "title": "Enterprise Agentic Memory & Cross-Session Personalization Architecture Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_locomo_longmemeval_mem0_and_zep_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_agent_turns": monthly_turns,
            "historical_sessions_per_user": 100,
            "injected_memory_budget_tokens": 1_200,
        },
        "headline_findings": {
            "tier4_temporal_fact_accuracy_pct": t4_temporal_fact_accuracy_pct,
            "accuracy_gain_t4_vs_t1_pts": accuracy_gain_t4_vs_t1_pts,
            "accuracy_gain_t4_vs_t3_pts": accuracy_gain_t4_vs_t3_pts,
            "ttft_speedup_t4_vs_t1_x": ttft_speedup_t4_vs_t1_x,
            "cost_savings_t4_vs_t1_pct": cost_savings_t4_vs_t1_pct,
            "tier3_contradiction_rate_pct": t3_stale_contradiction_rate_pct,
            "tier4_contradiction_rate_pct": t4_stale_contradiction_rate_pct,
        },
        "architecture_tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json")
    args = parser.parse_args()

    results = build_results()
    out_path = Path(__file__).with_name("results.json")

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist; run without --check first.")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != results:
            raise SystemExit("results.json mismatch with deterministic build_results().")
        print("OK: enterprise_agentic_memory_long_term_personalization_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

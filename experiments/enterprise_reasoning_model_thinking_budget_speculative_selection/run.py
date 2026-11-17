from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise Reasoning Model (DeepSeek-R1 / o3-mini / QwQ) Thinking Budget & Speculative Decoding (Round 385)
    # Workload: 10,000,000 enterprise queries/month (70% simple lookup/formatting, 20% moderate analysis, 10% complex math/code/policy reasoning).
    monthly_queries = 10_000_000

    # Tier 1: 100% Unbounded Deep Reasoning Mode (Every request runs full <think> without budget control)
    t1_avg_thinking_tokens = 3_250
    t1_task_accuracy_pct = 93.5
    t1_truncated_empty_answer_rate_pct = 0.8
    t1_avg_e2e_latency_sec = 24.50
    t1_monthly_cost_usd = 82_000

    # Tier 2: Naive Hard max_tokens Truncation (max_tokens=1024 cuts off inside <think> block!)
    t2_avg_thinking_tokens = 890
    t2_task_accuracy_pct = 74.0
    t2_truncated_empty_answer_rate_pct = 18.4  # Fails to close </think> on hard problems!
    t2_avg_e2e_latency_sec = 7.40
    t2_monthly_cost_usd = 24_800

    # Tier 3: 3-Gear Difficulty Router (70% Non-Thinking Fast V3 / 20% Low-Think / 10% High-Think R1)
    t3_avg_thinking_tokens = 680
    t3_task_accuracy_pct = 93.2
    t3_truncated_empty_answer_rate_pct = 1.2
    t3_avg_e2e_latency_sec = 4.20
    t3_monthly_cost_usd = 21_500

    # Tier 4: 3-Gear Router + Graceful </think> Budget-Forcing Stop + Native MTP / EAGLE-3 Speculative Decoding
    t4_avg_thinking_tokens = 620
    t4_task_accuracy_pct = 94.6
    t4_truncated_empty_answer_rate_pct = 0.0  # Graceful "</think>\nFinal Answer:" injection prevents cutoff
    t4_avg_e2e_latency_sec = 1.85
    t4_monthly_cost_usd = 12_800

    latency_speedup_t4_vs_t1_x = round(t1_avg_e2e_latency_sec / t4_avg_e2e_latency_sec, 2)
    cost_savings_t4_vs_t1_pct = round((t1_monthly_cost_usd - t4_monthly_cost_usd) / t1_monthly_cost_usd * 100.0, 1)
    thinking_token_reduction_pct = round((t1_avg_thinking_tokens - t4_avg_thinking_tokens) / t1_avg_thinking_tokens * 100.0, 1)

    tiers = [
        {
            "tier_id": "tier1_unbounded_full_reasoning_always",
            "name": "Tier 1: 100% 流量无脑全开深度思考 (简单查数也吐 3,250 个 <think> 废话 Token)",
            "avg_thinking_tokens_per_req": t1_avg_thinking_tokens,
            "task_accuracy_pct": t1_task_accuracy_pct,
            "truncated_mid_thought_rate_pct": t1_truncated_empty_answer_rate_pct,
            "avg_e2e_latency_sec": t1_avg_e2e_latency_sec,
            "monthly_cost_10m_queries_usd": t1_monthly_cost_usd,
            "status": "Severe Overthinking (24.5s & $82K/mo)",
        },
        {
            "tier_id": "tier2_naive_hard_max_tokens_cap",
            "name": "Tier 2: 粗暴硬截断 max_tokens=1024 (18.4% 复杂题在 <think> 内部被腰斩无答案)",
            "avg_thinking_tokens_per_req": t2_avg_thinking_tokens,
            "task_accuracy_pct": t2_task_accuracy_pct,
            "truncated_mid_thought_rate_pct": t2_truncated_empty_answer_rate_pct,
            "avg_e2e_latency_sec": t2_avg_e2e_latency_sec,
            "monthly_cost_10m_queries_usd": t2_monthly_cost_usd,
            "status": "Broken Output (18.4% Mid-Thought Cutoff)",
        },
        {
            "tier_id": "tier3_three_gear_reasoning_effort_router",
            "name": "Tier 3: 三档认知难度路由 (70% 免思考直答 + 20% 轻量思考 + 10% 深度推理)",
            "avg_thinking_tokens_per_req": t3_avg_thinking_tokens,
            "task_accuracy_pct": t3_task_accuracy_pct,
            "truncated_mid_thought_rate_pct": t3_truncated_empty_answer_rate_pct,
            "avg_e2e_latency_sec": t3_avg_e2e_latency_sec,
            "monthly_cost_10m_queries_usd": t3_monthly_cost_usd,
            "status": "Production Routing Baseline",
        },
        {
            "tier_id": "tier4_adaptive_budget_forcing_plus_mtp_speculative",
            "name": "Tier 4: 三档难度路由 + 优雅 </think> 预算收尾注入 + 原生 MTP/EAGLE-3 投机解码加速",
            "avg_thinking_tokens_per_req": t4_avg_thinking_tokens,
            "task_accuracy_pct": t4_task_accuracy_pct,
            "truncated_mid_thought_rate_pct": t4_truncated_empty_answer_rate_pct,
            "avg_e2e_latency_sec": t4_avg_e2e_latency_sec,
            "monthly_cost_10m_queries_usd": t4_monthly_cost_usd,
            "status": "Enterprise Reasoning Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_reasoning_model_thinking_budget_speculative_selection",
        "round": 385,
        "title": "Enterprise Reasoning Model (DeepSeek-R1 / o3-mini / QwQ) Thinking Budget & Speculative Decoding Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_reasoning_budget_forcing_and_mtp_eagle3_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_queries": monthly_queries,
            "fast_no_think_share_pct": 70.0,
            "medium_think_share_pct": 20.0,
            "deep_think_share_pct": 10.0,
        },
        "headline_findings": {
            "tier4_task_accuracy_pct": t4_task_accuracy_pct,
            "latency_speedup_t4_vs_t1_x": latency_speedup_t4_vs_t1_x,
            "cost_savings_t4_vs_t1_pct": cost_savings_t4_vs_t1_pct,
            "thinking_token_reduction_pct": thinking_token_reduction_pct,
            "tier2_truncated_rate_pct": t2_truncated_empty_answer_rate_pct,
            "tier4_truncated_rate_pct": t4_truncated_empty_answer_rate_pct,
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
        print("OK: enterprise_reasoning_model_thinking_budget_speculative_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

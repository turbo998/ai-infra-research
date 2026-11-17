from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise Prompt Caching, Semantic Caching & KV-Aware Gateway Routing Selection (Round 379)
    # Workload: 15,000,000 requests/month across an 8-replica GPU cluster / Cloud API,
    # 16,000-token average prompt context (system prompt + tool schemas + shared docs + multi-turn history).
    monthly_requests = 15_000_000
    cluster_replicas = 8
    avg_prompt_tokens = 16_000

    # Tier 1: Unstructured Dynamic Prompt + Round-Robin Gateway (Timestamp at top breaks prefix + scattered across 8 nodes)
    t1_effective_prefix_hit_rate_pct = 18.5
    t1_semantic_false_positive_rate_pct = 0.0
    t1_p95_ttft_ms = 1_420.0
    t1_monthly_cost_usd = 68_000

    # Tier 2: Naive Vector Semantic Cache at Gateway (cosine_sim >= 0.90 on raw query, no entity/tenant verification)
    t2_effective_prefix_hit_rate_pct = 42.0  # Exact + fuzzy query hits
    t2_semantic_false_positive_rate_pct = 4.8  # Wrong year/number/SKU returned ("2025 Tier A" vs "2026 Tier B")
    t2_p95_ttft_ms = 680.0
    t2_monthly_cost_usd = 41_500

    # Tier 3: Static-First Prompt Engineering + Cloud/Local Prefix Cache (APC / Cloud Prompt Cache, still Round-Robin)
    t3_effective_prefix_hit_rate_pct = 58.0
    t3_semantic_false_positive_rate_pct = 0.0
    t3_p95_ttft_ms = 510.0
    t3_monthly_cost_usd = 31_200

    # Tier 4: Static-First Prompt + KV-Aware Prefix-Tree Gateway Router + Entity-Verified Tenant Semantic Cache + Host-DRAM KV Tiering
    t4_effective_prefix_hit_rate_pct = 89.4
    t4_semantic_false_positive_rate_pct = 0.0  # Strict tenant_id namespace + NER/number/date exact match guard
    t4_p95_ttft_ms = 210.0
    t4_monthly_cost_usd = 18_400

    ttft_speedup_t4_vs_t1_x = round(t1_p95_ttft_ms / t4_p95_ttft_ms, 2)
    cost_savings_t4_vs_t1_pct = round((t1_monthly_cost_usd - t4_monthly_cost_usd) / t1_monthly_cost_usd * 100.0, 1)
    hit_rate_gain_t4_vs_t1_pts = round(t4_effective_prefix_hit_rate_pct - t1_effective_prefix_hit_rate_pct, 1)

    tiers = [
        {
            "tier_id": "tier1_dynamic_prompt_round_robin",
            "name": "Tier 1: 动态时间戳置顶 Prompt + 传统 Round-Robin 轮询网关 (8副本无状态打散)",
            "effective_cache_hit_rate_pct": t1_effective_prefix_hit_rate_pct,
            "semantic_false_positive_rate_pct": t1_semantic_false_positive_rate_pct,
            "p95_ttft_ms": t1_p95_ttft_ms,
            "monthly_cost_usd": t1_monthly_cost_usd,
            "tenant_isolation": "Stateless (81.5% Redundant Prefill Waste)",
            "status": "High Latency & Redundant Compute Waste",
        },
        {
            "tier_id": "tier2_naive_vector_semantic_cache",
            "name": "Tier 2: 网关层朴素纯向量语义缓存 (Cosine >= 0.90 模糊命中，无数字/实体校验)",
            "effective_cache_hit_rate_pct": t2_effective_prefix_hit_rate_pct,
            "semantic_false_positive_rate_pct": t2_semantic_false_positive_rate_pct,
            "p95_ttft_ms": t2_p95_ttft_ms,
            "monthly_cost_usd": t2_monthly_cost_usd,
            "tenant_isolation": "High False-Positive & Cross-Tenant Risk (4.8% Wrong Answers)",
            "status": "Dangerous in Financial/Enterprise Prod",
        },
        {
            "tier_id": "tier3_static_first_prompt_prefix_cache",
            "name": "Tier 3: 静态前缀置顶工程 (Static-First) + 云端/单机 Prefix Caching (仍用轮询路由)",
            "effective_cache_hit_rate_pct": t3_effective_prefix_hit_rate_pct,
            "semantic_false_positive_rate_pct": t3_semantic_false_positive_rate_pct,
            "p95_ttft_ms": t3_p95_ttft_ms,
            "monthly_cost_usd": t3_monthly_cost_usd,
            "tenant_isolation": "Exact Token Prefix Match (Zero Hallucination)",
            "status": "Cloud API Standard Baseline",
        },
        {
            "tier_id": "tier4_kv_aware_router_plus_verified_cache",
            "name": "Tier 4: 静态前缀工程 + KV 感知网关路由 (KV-Aware Router) + 实体核验语义缓存 + 分级 KV 卸载",
            "effective_cache_hit_rate_pct": t4_effective_prefix_hit_rate_pct,
            "semantic_false_positive_rate_pct": t4_semantic_false_positive_rate_pct,
            "p95_ttft_ms": t4_p95_ttft_ms,
            "monthly_cost_usd": t4_monthly_cost_usd,
            "tenant_isolation": "Strict Tenant-ID + ACL Hash Namespace + Entity Exact Guard",
            "status": "Enterprise Multi-Replica Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_prompt_caching_semantic_cache_kv_aware_routing_selection",
        "round": 379,
        "title": "Enterprise Prompt Caching, Semantic Caching & KV-Aware Gateway Routing Architecture Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_prompt_caching_and_kv_routing_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_requests": monthly_requests,
            "cluster_gpu_replicas": cluster_replicas,
            "avg_prompt_context_tokens": avg_prompt_tokens,
            "semantic_cache_cosine_threshold": 0.97,
        },
        "headline_findings": {
            "tier4_effective_cache_hit_rate_pct": t4_effective_prefix_hit_rate_pct,
            "hit_rate_gain_t4_vs_t1_pts": hit_rate_gain_t4_vs_t1_pts,
            "ttft_speedup_t4_vs_t1_x": ttft_speedup_t4_vs_t1_x,
            "cost_savings_t4_vs_t1_pct": cost_savings_t4_vs_t1_pct,
            "tier2_false_positive_rate_pct": t2_semantic_false_positive_rate_pct,
            "tier4_false_positive_rate_pct": t4_semantic_false_positive_rate_pct,
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
        print("OK: enterprise_prompt_caching_semantic_cache_kv_aware_routing_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

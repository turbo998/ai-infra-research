from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_agent_long_term_memory_consolidation_conflict_upsert_gdpr_selection",
        "title": "Enterprise Agent Long-Term Memory, Bi-Temporal Fact Upsert & Memory-Poisoning Firewall Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "active_enterprise_users": 500000,
            "monthly_recurring_sessions": 12000000,
            "avg_user_preference_updates_per_year": 18,
            "gdpr_erasure_sla_seconds": 5,
        },
        "tiers": [
            {
                "tier_id": "tier1_append_only_vector_chat_log_dump",
                "name": "Tier 1: Append-Only Vector Chat-Log Dump (No Fact Upsert, No Provenance Gate)",
                "stale_contradictory_memory_rate_pct": 29.4,
                "persistent_memory_poisoning_risk_pct": 14.2,
                "memory_retrieval_overhead_ms": 65.0,
                "monthly_memory_storage_and_token_tco_usd": 158000,
                "verdict": "Anti-Pattern: Recalls old & new facts together (29.4% conflict) & vulnerable to SpAIware poisoning",
            },
            {
                "tier_id": "tier2_unbounded_markdown_profile_prompt_stuffing",
                "name": "Tier 2: Unbounded Markdown Profile Appended to Every System Prompt",
                "stale_contradictory_memory_rate_pct": 16.8,
                "persistent_memory_poisoning_risk_pct": 9.5,
                "memory_retrieval_overhead_ms": 12.0,
                "monthly_memory_storage_and_token_tco_usd": 132000,
                "verdict": "Bloats every turn by +4,500 tokens & bleeds project-local context into unrelated tasks",
            },
            {
                "tier_id": "tier3_sync_inline_graph_llm_extraction",
                "name": "Tier 3: Synchronous Inline Graph-LLM Triple Extraction on User Path",
                "stale_contradictory_memory_rate_pct": 4.2,
                "persistent_memory_poisoning_risk_pct": 5.8,
                "memory_retrieval_overhead_ms": 2800.0,
                "monthly_memory_storage_and_token_tco_usd": 175000,
                "verdict": "Adds +2,800ms blocking turn latency & $175K/mo synchronous graph LLM extraction cost",
            },
            {
                "tier_id": "tier4_async_bitemporal_fact_upsert_provenance_firewall_scoped",
                "name": "Tier 4: Async Bi-Temporal (Subject,Predicate) Upsert + Provenance Firewall + 3-Scope Memory",
                "stale_contradictory_memory_rate_pct": 0.4,
                "persistent_memory_poisoning_risk_pct": 0.0,
                "memory_retrieval_overhead_ms": 8.0,
                "monthly_memory_storage_and_token_tco_usd": 19000,
                "verdict": "Recommended: 0.4% conflict rate, 0.0% SpAIware poisoning, 8ms lookup & -88.0% monthly TCO",
            },
        ],
        "headline_findings": {
            "contradictory_memory_reduction_pts": 29.0,
            "memory_poisoning_risk_pct": 0.0,
            "lookup_latency_ms": 8.0,
            "tco_savings_vs_tier1_pct": 88.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

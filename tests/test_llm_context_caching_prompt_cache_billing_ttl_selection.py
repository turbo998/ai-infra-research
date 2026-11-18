"""Unit tests for Enterprise LLM Context Caching, Prompt-Cache Billing Economics & KV TTL Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_context_caching_prompt_cache_billing_ttl_selection.run import (
    evaluate_llm_context_caching_prompt_cache_billing_ttl_matrix,
)


class TestLlmContextCachingPromptCacheBillingTtlSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_context_caching_prompt_cache_billing_ttl_matrix()
        self.assertEqual(data["experiment_id"], "llm_context_caching_prompt_cache_billing_ttl_selection")
        self.assertEqual(data["round"], 173)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_global_kv_radix_router_distributed_pool_append_only"]["cluster_cache_hit_rate_pct"],
            90.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_ttft_speedup_ratio"],
            15.0,
        )
        self.assertGreater(
            data["summary"]["tier4_monthly_savings_vs_tier1_usd"],
            250000.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_context_caching_prompt_cache_billing_ttl_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_context_caching_prompt_cache_billing_ttl_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

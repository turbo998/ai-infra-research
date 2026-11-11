"""Unit tests for Round 66 Cluster-Wide Prefix Cache, RadixAttention & KV-Aware Routing Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.prefix_cache_radix_kv_routing_eviction_selection.run import (
    evaluate_prefix_cache_radix_kv_routing_matrix,
)


class TestPrefixCacheRadixKvRoutingEvictionSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_prefix_cache_radix_kv_routing_matrix()
        self.assertEqual(
            data["experiment_id"],
            "prefix_cache_radix_kv_routing_eviction_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_dynamo_rational_decay_router_plus_tenant_slru"]["effective_cluster_kv_hit_rate_pct"],
            85.0,
        )
        self.assertLess(
            tiers["tier4_dynamo_rational_decay_router_plus_tenant_slru"]["p99_ttft_ms"],
            200.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_p99_ttft_speedup_ratio"],
            15.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "prefix_cache_radix_kv_routing_eviction_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_prefix_cache_radix_kv_routing_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

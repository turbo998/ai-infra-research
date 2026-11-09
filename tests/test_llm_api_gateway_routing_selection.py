from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.llm_api_gateway_routing_selection.run import RESULTS_PATH, compute_model


class TestLlmApiGatewayRoutingSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_gateway_routing_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_stateless_round_robin"]
        t2 = tiers["tier2_token_ratelimit_exact_cache"]
        t3 = tiers["tier3_kv_prefix_semantic_cache"]
        t4 = tiers["tier4_cascade_router_slo_breaker"]

        self.assertEqual(t1["kv_prefix_hit_rate_pct"], 8.1)
        self.assertEqual(t1["monthly_cost_usd"], 142000.00)
        self.assertEqual(t2["semantic_or_exact_cache_hit_pct"], 8.5)
        self.assertEqual(t3["kv_prefix_hit_rate_pct"], 64.2)
        self.assertEqual(t3["p99_ttft_ms"], 310.0)
        self.assertEqual(t4["p99_ttft_ms"], 185.0)
        self.assertEqual(t4["quality_retention_pct"], 99.2)
        self.assertEqual(t4["monthly_cost_usd"], 26980.00)
        self.assertEqual(t4["cost_savings_pct"], 81.0)


if __name__ == "__main__":
    unittest.main()

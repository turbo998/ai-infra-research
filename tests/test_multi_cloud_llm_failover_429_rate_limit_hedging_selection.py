from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.multi_cloud_llm_failover_429_rate_limit_hedging_selection.run import (
    evaluate_multi_cloud_llm_failover_hedging_tiers,
)


class TestMultiCloudLlmFailover429RateLimitHedgingSelection(unittest.TestCase):
    def test_provenance_and_multicloud_failover_invariants(self) -> None:
        data = evaluate_multi_cloud_llm_failover_hedging_tiers()
        self.assertEqual(
            data["experiment_id"],
            "multi_cloud_llm_failover_429_rate_limit_hedging_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier1_p99_speedup_x"], 38.8)
        self.assertEqual(summary["tier4_vs_tier2_p99_speedup_x"], 9.14)
        self.assertEqual(summary["tier4_availability_pct"], 99.995)
        self.assertEqual(summary["tier4_mid_stream_waste_pct"], 0.6)
        self.assertEqual(summary["tier4_vs_tier1_annual_savings_usd"], 2640000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "multi_cloud_llm_failover_429_rate_limit_hedging_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            on_disk, evaluate_multi_cloud_llm_failover_hedging_tiers()
        )


if __name__ == "__main__":
    unittest.main()

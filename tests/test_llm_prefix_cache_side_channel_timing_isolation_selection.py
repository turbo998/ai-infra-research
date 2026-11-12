from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.llm_prefix_cache_side_channel_timing_isolation_selection.run import (
    evaluate_prefix_cache_side_channel_isolation_tiers,
)


class TestLlmPrefixCacheSideChannelTimingIsolationSelection(unittest.TestCase):
    def test_provenance_and_prefix_cache_security_invariants(self) -> None:
        data = evaluate_prefix_cache_side_channel_isolation_tiers()
        self.assertEqual(
            data["experiment_id"],
            "llm_prefix_cache_side_channel_timing_isolation_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_attack_success_pct"], 0.0)
        self.assertEqual(summary["tier1_attack_success_pct"], 98.4)
        self.assertEqual(summary["tier4_vs_tier2_ttft_speedup_x"], 4.26)
        self.assertEqual(summary["tier4_vs_tier2_throughput_gain_x"], 3.11)
        self.assertEqual(summary["tier4_vs_tier2_annual_savings_usd"], 1512000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "llm_prefix_cache_side_channel_timing_isolation_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            on_disk, evaluate_prefix_cache_side_channel_isolation_tiers()
        )


if __name__ == "__main__":
    unittest.main()

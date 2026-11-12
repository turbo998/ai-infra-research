from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.moe_expert_load_balancing_eplb_auxfree_selection.run import (
    evaluate_moe_eplb_auxfree_routing_tiers,
)


class TestMoeExpertLoadBalancingEplbAuxfreeSelection(unittest.TestCase):
    def test_provenance_and_moe_eplb_invariants(self) -> None:
        data = evaluate_moe_eplb_auxfree_routing_tiers()
        self.assertEqual(
            data["experiment_id"],
            "moe_expert_load_balancing_eplb_auxfree_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier2_goodput_gain_x"], 3.61)
        self.assertEqual(summary["tier4_vs_tier2_p99_speedup_x"], 3.82)
        self.assertEqual(summary["tier4_vs_tier1_accuracy_gain_pt"], 10.3)
        self.assertEqual(summary["tier4_skew_ratio_x"], 1.08)
        self.assertEqual(summary["tier4_vs_tier2_annual_savings_usd"], 798000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "moe_expert_load_balancing_eplb_auxfree_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_moe_eplb_auxfree_routing_tiers())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Round 372 Enterprise Multi-Region High Availability (HA/DR) & Cross-Cloud LLM Routing Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_multi_region_ha_dr_cross_cloud_routing_selection.run import (
    evaluate_enterprise_multi_region_ha_dr_cross_cloud_routing,
)


class TestEnterpriseMultiRegionHaDrCrossCloudRoutingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_multi_region_ha_dr_cross_cloud_routing()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_multi_region_ha_dr_cross_cloud_routing_selection",
        )
        self.assertEqual(data["round"], 372)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreaterEqual(
            tiers["tier4_multi_cloud_active_active_plus_p95_tail_hedging"]["effective_availability_pct"],
            99.99,
        )
        self.assertLess(
            tiers["tier4_multi_cloud_active_active_plus_p95_tail_hedging"]["monthly_tco_usd"],
            tiers["tier2_active_passive_duplicate_idle_standby_cluster"]["monthly_tco_usd"],
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_multi_region_ha_dr_cross_cloud_routing_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_multi_region_ha_dr_cross_cloud_routing()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

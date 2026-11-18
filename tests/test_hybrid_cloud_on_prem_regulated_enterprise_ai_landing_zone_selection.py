"""Unit tests for Hybrid Cloud & On-Premises Regulated Enterprise AI Landing Zone Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.hybrid_cloud_on_prem_regulated_enterprise_ai_landing_zone_selection.run import (
    evaluate_hybrid_cloud_on_prem_regulated_enterprise_ai_landing_zone,
)


class TestHybridCloudOnPremRegulatedEnterpriseAiLandingZoneSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_hybrid_cloud_on_prem_regulated_enterprise_ai_landing_zone()
        self.assertEqual(
            data["experiment_id"],
            "hybrid_cloud_on_prem_regulated_enterprise_ai_landing_zone_selection",
        )
        self.assertEqual(data["round"], 366)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertEqual(
            tiers["tier4_hybrid_data_classification_reversible_pii_vault_router"]["pii_compliance_violation_pct"],
            0.0,
        )
        self.assertGreater(
            tiers["tier4_hybrid_data_classification_reversible_pii_vault_router"]["complex_reasoning_quality_pct"],
            tiers["tier2_100pct_on_premises_airgapped_gpu_island"]["complex_reasoning_quality_pct"],
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_on_prem_tco_savings_pct"],
            60.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "hybrid_cloud_on_prem_regulated_enterprise_ai_landing_zone_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_hybrid_cloud_on_prem_regulated_enterprise_ai_landing_zone()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

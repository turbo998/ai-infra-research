"""Unit tests for Cloud Managed vs. Self-Hosted LLM Deployment Mode & TCO Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.cloud_llm_deployment_paygo_ptu_aks_kaito_tco_selection.run import (
    evaluate_cloud_llm_deployment_paygo_ptu_aks_kaito_tco,
)


class TestCloudLlmDeploymentPaygoPtuAksKaitoTcoSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_cloud_llm_deployment_paygo_ptu_aks_kaito_tco()
        self.assertEqual(data["experiment_id"], "cloud_llm_deployment_paygo_ptu_aks_kaito_tco_selection")
        self.assertEqual(data["round"], 361)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier4_hybrid_base_ptu_or_aks_plus_burst_paygo"]["monthly_tco_usd"],
            tiers["tier2_full_peak_static_ptu_provisioned"]["monthly_tco_usd"],
        )
        self.assertEqual(
            tiers["tier4_hybrid_base_ptu_or_aks_plus_burst_paygo"]["http_429_throttling_rate_pct"],
            0.0,
        )
        self.assertGreater(
            data["summary"]["tier4_cost_savings_vs_tier2_peak_ptu_pct"],
            70.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "cloud_llm_deployment_paygo_ptu_aks_kaito_tco_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_cloud_llm_deployment_paygo_ptu_aks_kaito_tco()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

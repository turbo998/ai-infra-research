"""Unit tests for Round 365 Multi-Team Enterprise Internal MaaS Platform & FinOps Chargeback Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_internal_maas_platform_finops_chargeback_selection.run import (
    evaluate_enterprise_internal_maas_platform_finops_chargeback,
)


class TestEnterpriseInternalMaasPlatformFinopsChargebackSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_internal_maas_platform_finops_chargeback()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_internal_maas_platform_finops_chargeback_selection",
        )
        self.assertEqual(data["round"], 365)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier4_enterprise_maas_finops_chargeback_qos_tidal_backfill"]["monthly_tco_usd"],
            tiers["tier1_departmental_gpu_silos_dedicated_instances"]["monthly_tco_usd"],
        )
        self.assertGreater(
            tiers["tier4_enterprise_maas_finops_chargeback_qos_tidal_backfill"]["avg_gpu_utilization_pct"],
            80.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_tco_savings_pct"],
            75.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_internal_maas_platform_finops_chargeback_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_internal_maas_platform_finops_chargeback()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

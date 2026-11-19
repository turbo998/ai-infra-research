from __future__ import annotations

import unittest
from experiments.enterprise_text2sql_semantic_layer_governed_bi_agent_selection.run import build_payload


class TestEnterpriseText2SqlSemanticLayerGovernedBiAgentSelection(unittest.TestCase):
    def test_tier4_highest_kpi_accuracy_full_consistency_zero_runaway_scans_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["business_kpi_execution_accuracy_pct"], 97.0)
        self.assertGreater(
            tier4["business_kpi_execution_accuracy_pct"],
            tier2["business_kpi_execution_accuracy_pct"] + 20.0,
        )
        self.assertEqual(tier4["metric_definition_consistency_across_users_pct"], 100.0)
        self.assertEqual(tier4["runaway_full_table_scan_warehouse_risk_pct"], 0.0)
        self.assertEqual(tier4["row_level_security_and_column_masking_compliance_pct"], 100.0)
        self.assertLess(
            tier4["monthly_data_warehouse_and_rework_tco_usd"],
            tier1["monthly_data_warehouse_and_rework_tco_usd"] * 0.20,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()

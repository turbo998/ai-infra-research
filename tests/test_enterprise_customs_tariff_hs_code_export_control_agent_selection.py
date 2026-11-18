import unittest
from experiments.enterprise_customs_tariff_hs_code_export_control_agent_selection.run import build_payload


class TestEnterpriseCustomsTariffHsCodeExportControlAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_hs_accuracy_with_zero_sanctions_misses(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_customs_broker_and_spreadsheet_roo"]
        t2 = tiers["tier2_prompt_only_llm_freeform_hs_code_and_screening"]
        t4 = tiers["tier4_wco_gri_chapter_note_tree_plus_bom_rvc_engine_plus_ofac_50pct_graph_agent"]

        self.assertGreater(t4["ten_digit_hs_classification_accuracy_pct"], 95.0)
        self.assertEqual(t4["sanctions_or_eccn_compliance_miss_rate_pct"], 0.0)
        self.assertGreater(t2["sanctions_or_eccn_compliance_miss_rate_pct"], 10.0)
        self.assertLessEqual(t4["declaration_clearance_prep_minutes"], 3.0)
        self.assertLess(t4["monthly_brokerage_and_penalty_cost_usd"], t1["monthly_brokerage_and_penalty_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

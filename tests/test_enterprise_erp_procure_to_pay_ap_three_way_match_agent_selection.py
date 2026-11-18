import unittest
from experiments.enterprise_erp_procure_to_pay_ap_three_way_match_agent_selection.run import build_payload


class TestEnterpriseErpProcureToPayApThreeWayMatchAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_touchless_stp_with_zero_bec_or_overpayment_errors(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_template_ocr_and_manual_ap_clerk_matching"]
        t2 = tiers["tier2_prompt_only_vlm_direct_sap_miro_posting"]
        t4 = tiers["tier4_vlm_uom_normalizer_plus_deterministic_3way_tolerance_engine_plus_bec_bank_gate"]

        self.assertGreater(t4["touchless_three_way_match_stp_rate_pct"], 85.0)
        self.assertEqual(t4["overpayment_duplicate_or_bec_fraud_rate_pct"], 0.0)
        self.assertGreater(t2["overpayment_duplicate_or_bec_fraud_rate_pct"], 5.0)
        self.assertLessEqual(t4["cost_per_processed_invoice_usd"], 1.80)
        self.assertLess(t4["monthly_ap_processing_and_leakage_cost_usd"], t1["monthly_ap_processing_and_leakage_cost_usd"] * 0.15)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

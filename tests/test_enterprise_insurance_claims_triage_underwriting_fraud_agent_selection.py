import unittest
from experiments.enterprise_insurance_claims_triage_underwriting_fraud_agent_selection.run import build_payload


class TestEnterpriseInsuranceClaimsTriageUnderwritingFraudAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_fraud_recall_and_actuarial_payout_accuracy(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_claims_adjuster_and_legacy_rule_engine"]
        t2 = tiers["tier2_end_to_end_vlm_prompt_adjudication_and_mental_payout"]
        t4 = tiers["tier4_pixel_forensics_gnn_fraud_plus_policy_ast_actuarial_engine_siu_agent"]

        self.assertGreater(t4["straight_through_processing_stp_rate_pct"], 80.0)
        self.assertGreater(t4["fraud_and_forged_invoice_detection_recall_pct"], 95.0)
        self.assertGreater(t4["payout_calculation_and_clause_accuracy_pct"], 99.0)
        self.assertLess(t2["payout_calculation_and_clause_accuracy_pct"], 70.0)
        self.assertLess(t4["monthly_claims_ops_and_fraud_leakage_cost_usd"], t1["monthly_claims_ops_and_fraud_leakage_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

import unittest
from experiments.enterprise_esg_carbon_accounting_scope3_csrd_audit_agent_selection.run import build_payload


class TestEnterpriseEsgCarbonAccountingScope3CsrdAuditAgentSelection(unittest.TestCase):
    def test_tier4_achieves_zero_factor_hallucination_and_full_isae3000_traceability(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_excel_spend_based_eeio_and_consultants"]
        t2 = tiers["tier2_prompt_only_llm_memorized_factors_and_mental_math"]
        t4 = tiers["tier4_multimodal_utility_bom_extractor_plus_ecoinvent_hierarchy_plus_isae3000_lineage_agent"]

        self.assertGreater(t4["ghg_audit_assurance_accuracy_pct"], 98.5)
        self.assertEqual(t4["emission_factor_or_unit_hallucination_rate_pct"], 0.0)
        self.assertEqual(t4["audit_evidence_traceability_completeness_pct"], 100.0)
        self.assertGreater(t2["emission_factor_or_unit_hallucination_rate_pct"], 20.0)
        self.assertLess(t4["monthly_esg_reporting_and_cbam_overpayment_cost_usd"], t1["monthly_esg_reporting_and_cbam_overpayment_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

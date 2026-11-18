import unittest
from experiments.enterprise_wealth_management_investment_research_compliance_agent_selection.run import build_payload


class TestEnterpriseWealthManagementInvestmentResearchComplianceAgentSelection(unittest.TestCase):
    def test_tier4_achieves_zero_numerical_hallucination_and_highest_accuracy(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_analyst_ctrl_f_excel_modeling"]
        t2 = tiers["tier2_naive_vector_chunk_rag_llm_mental_math"]
        t4 = tiers["tier4_xbrl_sql_store_plus_hierarchical_rag_plus_python_sandbox_sec_gate"]

        self.assertGreaterEqual(t4["financial_metric_and_ratio_accuracy_pct"], 99.0)
        self.assertEqual(t4["numerical_hallucination_or_unit_mismatch_rate_pct"], 0.0)
        self.assertLess(t2["financial_metric_and_ratio_accuracy_pct"], 70.0)
        self.assertLess(t4["monthly_research_and_compliance_cost_usd"], t1["monthly_research_and_compliance_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

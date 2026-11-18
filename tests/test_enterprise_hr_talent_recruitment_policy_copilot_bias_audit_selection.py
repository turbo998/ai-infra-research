import unittest
from experiments.enterprise_hr_talent_recruitment_policy_copilot_bias_audit_selection.run import build_payload


class TestEnterpriseHrTalentRecruitmentPolicyCopilotBiasAuditSelection(unittest.TestCase):
    def test_tier4_eliminates_bias_and_policy_leaks_while_maximizing_ndcg(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_ats_exact_keyword_boolean_filter"]
        t2 = tiers["tier2_unmasked_raw_resume_holistic_llm_scoring"]
        t4 = tiers["tier4_pii_blind_esco_ontology_binary_rubric_eeoc_gate_rbac_copilot"]

        self.assertGreater(t4["shortlist_hire_quality_ndcg_at_10_pct"], 90.0)
        self.assertEqual(t4["eeoc_adverse_impact_violation_rate_pct"], 0.0)
        self.assertEqual(t4["cross_country_hr_policy_leak_rate_pct"], 0.0)
        self.assertGreater(t2["eeoc_adverse_impact_violation_rate_pct"], 20.0)
        self.assertLess(t4["monthly_recruiting_and_hr_ops_cost_usd"], t1["monthly_recruiting_and_hr_ops_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

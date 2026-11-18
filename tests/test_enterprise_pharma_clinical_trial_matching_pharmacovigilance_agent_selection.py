import unittest
from experiments.enterprise_pharma_clinical_trial_matching_pharmacovigilance_agent_selection.run import build_payload


class TestEnterprisePharmaClinicalTrialMatchingPharmacovigilanceAgentSelection(unittest.TestCase):
    def test_tier4_achieves_zero_exclusion_safety_misses_and_high_meddra_accuracy(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_cra_chart_review_and_manual_meddra_coding"]
        t2 = tiers["tier2_monolithic_prompt_llm_raw_ehr_and_freeform_meddra"]
        t4 = tiers["tier4_atomic_logic_compiler_fhir_labs_plus_meddra_hierarchy_21cfr11_agent"]

        self.assertGreater(t4["trial_eligibility_f1_accuracy_pct"], 95.0)
        self.assertEqual(t4["exclusion_criteria_safety_miss_rate_pct"], 0.0)
        self.assertGreater(t4["meddra_pt_coding_accuracy_pct"], 97.0)
        self.assertGreater(t2["exclusion_criteria_safety_miss_rate_pct"], 15.0)
        self.assertLess(t4["monthly_clinical_and_pv_ops_cost_usd"], t1["monthly_clinical_and_pv_ops_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

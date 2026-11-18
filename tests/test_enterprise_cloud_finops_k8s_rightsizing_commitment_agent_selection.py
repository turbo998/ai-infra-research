import unittest
from experiments.enterprise_cloud_finops_k8s_rightsizing_commitment_agent_selection.run import build_payload


class TestEnterpriseCloudFinopsK8sRightsizingCommitmentAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_savings_with_zero_oomkilled_or_stranded_ri(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_monthly_excel_finops_review_and_manual_ri_buying"]
        t2 = tiers["tier2_naive_avg_utilization_llm_auto_rightsizing_and_3yr_ri"]
        t4 = tiers["tier4_focus_semantic_layer_plus_jvm_aware_p999_rightsizing_plus_lp_commitment_ladder_agent"]

        self.assertGreater(t4["effective_cloud_bill_savings_pct"], 35.0)
        self.assertEqual(t4["rightsizing_oomkilled_or_stranded_commitment_rate_pct"], 0.0)
        self.assertGreater(t2["rightsizing_oomkilled_or_stranded_commitment_rate_pct"], 12.0)
        self.assertLessEqual(t4["anomaly_detection_to_remediation_hours"], 1.0)
        self.assertLess(t4["net_monthly_waste_and_finops_ops_cost_usd"], t1["net_monthly_waste_and_finops_ops_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

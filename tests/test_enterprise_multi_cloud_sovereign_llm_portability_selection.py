import unittest
from experiments.enterprise_multi_cloud_sovereign_llm_portability_selection.run import build_Payload


class TestEnterpriseMultiCloudSovereignLlmPortabilitySelection(unittest.TestCase):
    def test_tier4_achieves_100pct_sovereignty_and_high_failover_consistency(self) -> None:
        data = build_Payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_single_cloud_proprietary_sdk_lockin"]
        t2 = tiers["tier2_naive_url_swap_multi_cloud_proxy"]
        t3 = tiers["tier3_uniform_dedicated_gpu_clusters_everywhere"]
        t4 = tiers["tier4_geo_sovereign_gateway_compiled_prompt_adapters_hybrid_anchor"]

        self.assertEqual(t4["sovereign_compliance_pct"], 100.0)
        self.assertGreater(t4["failover_behavioral_consistency_pct"], t2["failover_behavioral_consistency_pct"] + 25.0)
        self.assertLess(t4["migration_effort_person_weeks"], t1["migration_effort_person_weeks"] * 0.1)
        self.assertLess(t4["monthly_global_cost_usd"], t3["monthly_global_cost_usd"] * 0.4)

    def test_provenance_Explicit(self) -> None:
        data = build_Payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

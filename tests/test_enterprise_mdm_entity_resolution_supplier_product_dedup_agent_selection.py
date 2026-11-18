import unittest
from experiments.enterprise_mdm_entity_resolution_supplier_product_dedup_agent_selection.run import build_payload


class TestEnterpriseMdmEntityResolutionSupplierProductDedupAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_f1_with_zero_false_positive_bad_merges(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_edit_distance_rules_and_manual_steward_merge"]
        t2 = tiers["tier2_dense_vector_threshold_auto_merge_and_naive_llm"]
        t4 = tiers["tier4_hybrid_lsh_vector_blocking_plus_spec_attribute_extractor_plus_tax_duns_hard_gate_agent"]

        self.assertGreater(t4["entity_resolution_f1_score_pct"], 96.0)
        self.assertEqual(t4["false_positive_bad_merge_rate_pct"], 0.0)
        self.assertGreater(t2["false_positive_bad_merge_rate_pct"], 10.0)
        self.assertGreater(t4["candidate_pair_reduction_ratio_pct"], 99.9)
        self.assertLess(t4["monthly_steward_and_compute_cost_usd"], t1["monthly_steward_and_compute_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

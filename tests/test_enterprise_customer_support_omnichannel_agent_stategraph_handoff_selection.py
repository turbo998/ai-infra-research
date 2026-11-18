import unittest
from experiments.enterprise_customer_support_omnichannel_agent_stategraph_handoff_selection.run import build_payload


class TestEnterpriseCustomerSupportOmnichannelAgentStategraphHandoffSelection(unittest.TestCase):
    def test_tier4_maximizes_fcr_with_zero_unauthorized_refunds(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_rigid_keyword_decision_tree_bot"]
        t2 = tiers["tier2_unconstrained_prompt_llm_direct_refund_tool"]
        t4 = tiers["tier4_effective_date_rag_plus_stategraph_policy_gate_plus_warm_handoff"]

        self.assertGreater(t4["first_contact_resolution_fcr_pct"], 80.0)
        self.assertEqual(t4["unauthorized_refund_or_policy_violation_rate_pct"], 0.0)
        self.assertGreater(t2["unauthorized_refund_or_policy_violation_rate_pct"], 10.0)
        self.assertLessEqual(t4["human_handoff_repeat_explanation_time_sec"], 25.0)
        self.assertLess(t4["monthly_support_and_refund_leakage_cost_usd"], t1["monthly_support_and_refund_leakage_cost_usd"] * 0.25)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

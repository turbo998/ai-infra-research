import unittest
from experiments.enterprise_financial_fraud_aml_agent_selection.run import build_Payload


class TestEnterpriseFinancialFraudAmlAgentSelection(unittest.TestCase):
    def test_tier4_meets_payment_latency_sla_and_slashes_fpr(self) -> None:
        data = build_Payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_sql_velocity_rules_manual_aml"]
        t2 = tiers["tier2_sync_inline_llm_on_payment_switch"]
        t4 = tiers["tier4_hot_gnn_xgboost_warm_8b_greyzone_cold_graph_aml_copilot"]

        self.assertLess(t4["payment_auth_p99_latency_ms"], 20.0)
        self.assertGreater(t2["payment_auth_p99_latency_ms"], 1000.0)
        self.assertLess(t4["false_positive_alert_rate_pct"], 20.0)
        self.assertLess(t4["aml_case_review_minutes"], 5.0)
        self.assertLess(t4["monthly_total_cost_usd"], t1["monthly_total_cost_usd"] * 0.2)

    def test_provenance_Explicit(self) -> None:
        data = build_Payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

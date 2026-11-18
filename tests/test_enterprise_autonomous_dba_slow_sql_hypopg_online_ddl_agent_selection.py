import unittest
from experiments.enterprise_autonomous_dba_slow_sql_hypopg_online_ddl_agent_selection.run import build_payload


class TestEnterpriseAutonomousDbaSlowSqlHypopgOnlineDdlAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_latency_reduction_with_zero_ddl_lock_outages(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_dba_firefighting_and_heuristic_indexing"]
        t2 = tiers["tier2_prompt_only_llm_direct_production_ddl_execution"]
        t4 = tiers["tier4_hypopg_virtual_index_verifier_plus_sqlglot_equivalence_plus_online_ddl_gate"]

        self.assertGreater(t4["slow_sql_p99_latency_reduction_pct"], 90.0)
        self.assertEqual(t4["ddl_table_lock_outage_or_plan_regression_rate_pct"], 0.0)
        self.assertGreater(t2["ddl_table_lock_outage_or_plan_regression_rate_pct"], 15.0)
        self.assertLessEqual(t4["mean_time_to_resolve_slow_sql_minutes"], 10.0)
        self.assertLess(t4["monthly_db_overprovisioning_and_dba_ops_cost_usd"], t1["monthly_db_overprovisioning_and_dba_ops_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

import unittest
from experiments.enterprise_telecom_5g_network_noc_rca_digital_twin_agent_selection.run import build_payload


class TestEnterpriseTelecom5gNetworkNocRcaDigitalTwinAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_rca_accuracy_with_zero_config_outages(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_noc_static_rules_and_manual_cli_triage"]
        t2 = tiers["tier2_raw_alarm_stream_llm_with_direct_netconf_cli_push"]
        t4 = tiers["tier4_cross_layer_topology_graph_rca_plus_yang_digital_twin_canary_agent"]

        self.assertGreater(t4["cross_layer_root_cause_localization_accuracy_pct"], 94.0)
        self.assertEqual(t4["config_change_network_outage_or_syntax_error_rate_pct"], 0.0)
        self.assertGreater(t2["config_change_network_outage_or_syntax_error_rate_pct"], 10.0)
        self.assertLessEqual(t4["mean_time_to_restore_service_minutes"], 10.0)
        self.assertLess(t4["monthly_noc_ops_and_sla_penalty_cost_usd"], t1["monthly_noc_ops_and_sla_penalty_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

import unittest
from experiments.enterprise_cybersecurity_soc_threat_hunting_soar_copilot_selection.run import build_payload


class TestEnterpriseCybersecuritySocThreatHuntingSoarCopilotSelection(unittest.TestCase):
    def test_tier4_eliminates_log_injection_and_minimizes_mttc(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_siem_sigma_rules_and_manual_l1_triage"]
        t2 = tiers["tier2_raw_log_prompt_llm_with_autonomous_soar_isolation"]
        t4 = tiers["tier4_ocsf_entity_attack_graph_plus_kql_sandbox_plus_soar_blast_radius_gate"]

        self.assertGreater(t4["attack_chain_attribution_accuracy_pct"], 95.0)
        self.assertEqual(t4["critical_asset_false_isolation_or_log_injection_rate_pct"], 0.0)
        self.assertGreater(t2["critical_asset_false_isolation_or_log_injection_rate_pct"], 10.0)
        self.assertLessEqual(t4["mean_time_to_investigate_and_contain_minutes"], 10.0)
        self.assertLess(t4["monthly_soc_ops_and_breach_exposure_cost_usd"], t1["monthly_soc_ops_and_breach_exposure_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

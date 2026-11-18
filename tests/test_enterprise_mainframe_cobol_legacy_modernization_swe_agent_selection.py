import unittest
from experiments.enterprise_mainframe_cobol_legacy_modernization_swe_agent_selection.run import build_payload


class TestEnterpriseMainframeCobolLegacyModernizationSweAgentSelection(unittest.TestCase):
    def test_tier4_achieves_penny_exact_equivalence_and_highest_velocity(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_mechanical_line_by_line_transpiler_and_manual_rewrite"]
        t2 = tiers["tier2_file_by_file_prompt_llm_direct_java_python_translation"]
        t4 = tiers["tier4_program_dependency_graph_spec_first_bigdecimal_shadow_replay_agent"]

        self.assertGreaterEqual(t4["functional_and_penny_equivalence_pass_rate_pct"], 99.8)
        self.assertEqual(t4["comp3_decimal_or_redefines_regression_rate_pct"], 0.0)
        self.assertGreater(t2["comp3_decimal_or_redefines_regression_rate_pct"], 15.0)
        self.assertGreaterEqual(t4["migration_velocity_kloc_per_month"], 120.0)
        self.assertLess(t4["monthly_modernization_and_mips_dual_run_cost_usd"], t1["monthly_modernization_and_mips_dual_run_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

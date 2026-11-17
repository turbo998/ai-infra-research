import unittest

from experiments.enterprise_ai_agent_code_execution_sandbox_microvm_selection.run import build_results


class TestEnterpriseAiAgentCodeSandboxSelection(unittest.TestCase):
    def test_microvm_snapshot_and_isolation_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 395)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_ready_latency_ms"], 115.0)
        self.assertEqual(h["startup_speedup_t4_vs_t2_x"], 160.9)
        self.assertEqual(h["startup_speedup_t4_vs_t3_x"], 5.91)
        self.assertEqual(h["cost_savings_t4_vs_t2_pct"], 84.1)
        self.assertEqual(h["tier4_kernel_isolation_pct"], 100.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["ready_with_pandas_latency_ms"], tiers[0]["ready_with_pandas_latency_ms"])
        self.assertEqual(tiers[3]["kernel_isolation_score_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()

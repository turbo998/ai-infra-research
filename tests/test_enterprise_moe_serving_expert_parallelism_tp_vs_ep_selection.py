import unittest

from experiments.enterprise_moe_serving_expert_parallelism_tp_vs_ep_selection.run import build_results


class TestEnterpriseMoeServingExpertParallelismSelection(unittest.TestCase):
    def test_moe_ep_and_eplb_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 396)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_per_gpu_tok_s"], 738)
        self.assertEqual(h["throughput_gain_t4_vs_t1_x"], 4.99)
        self.assertEqual(h["throughput_gain_t4_vs_t3_x"], 1.59)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 60.4)
        self.assertEqual(h["tier3_straggler_ratio"], 3.4)
        self.assertEqual(h["tier4_straggler_ratio"], 1.08)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["per_gpu_throughput_tok_s"], tiers[2]["per_gpu_throughput_tok_s"])
        self.assertLess(tiers[3]["p99_tpot_ms"], 30.0)


if __name__ == "__main__":
    unittest.main()

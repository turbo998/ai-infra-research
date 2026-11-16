import unittest
from experiments.llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection.run import evaluate_flashattention_varlen_skew_scheduler


class TestFlashAttentionVarlenSkewScheduler(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_flashattention_varlen_skew_scheduler()
        self.assertEqual(res["round"], 321)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_persistent_clc_scheduler_lpt_causal_cost_plus_split_kv"]["kernel_latency_ms"],
            4.0,
        )
        self.assertGreater(
            tiers["tier4_persistent_clc_scheduler_lpt_causal_cost_plus_split_kv"]["average_sm_utilization_pct"],
            85.0,
        )


if __name__ == "__main__":
    unittest.main()

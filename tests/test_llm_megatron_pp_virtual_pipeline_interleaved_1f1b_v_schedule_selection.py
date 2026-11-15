import unittest
from experiments.llm_megatron_pp_virtual_pipeline_interleaved_1f1b_v_schedule_selection.run import evaluate_pp_schedules_memory_and_bubble


class TestPpSchedulesMemoryAndBubble(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_pp_schedules_memory_and_bubble()
        self.assertEqual(res["round"], 290)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier2_interleaved_1f1b_vpp4_p2p_and_memory_inflation"]["stage0_peak_activation_gb"],
            tiers["tier4_zb_v_dualpipe_bidirectional_split_bw_uniform_memory"]["stage0_peak_activation_gb"] * 2.0,
        )
        self.assertLess(
            tiers["tier4_zb_v_dualpipe_bidirectional_split_bw_uniform_memory"]["pipeline_bubble_pct"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()

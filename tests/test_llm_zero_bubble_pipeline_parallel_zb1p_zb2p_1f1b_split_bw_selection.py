import unittest
from experiments.llm_zero_bubble_pipeline_parallel_zb1p_zb2p_1f1b_split_bw_selection.run import (
    evaluate_zero_bubble_pp_tiers,
)


class TestZeroBubblePpSelection(unittest.TestCase):
    def test_zero_bubble_pp_tiers(self) -> None:
        data = evaluate_zero_bubble_pp_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t2 = tiers["tier2_standard_1f1b_monolithic_backward"]
        t4 = tiers["tier4_zero_bubble_pp_split_backward_b_and_w_zbh1_zbh2"]
        self.assertGreater(t2["pipeline_bubble_pct"], 20.0)
        self.assertLess(t4["pipeline_bubble_pct"], 2.0)
        self.assertEqual(t4["p2p_comm_volume_multiplier_x"], 1.0)
        self.assertGreater(t4["step_throughput_tflops"], 520.0)


if __name__ == "__main__":
    unittest.main()

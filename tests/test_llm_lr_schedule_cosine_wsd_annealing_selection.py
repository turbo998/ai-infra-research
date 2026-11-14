import unittest
from experiments.llm_lr_schedule_cosine_wsd_annealing_selection.run import (
    evaluate_lr_schedule_tiers,
)


class TestLrScheduleWsdSelection(unittest.TestCase):
    def test_tiers_and_wsd_advantage(self) -> None:
        data = evaluate_lr_schedule_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_cosine_decay_rewarmup_shock"]
        t4 = tiers["tier4_wsd_1_minus_sqrt_decay_plus_hq_annealing_ema"]
        self.assertGreater(t1["wasted_recovery_tokens_b"], 1000.0)
        self.assertEqual(t4["wasted_recovery_tokens_b"], 0.0)
        self.assertGreaterEqual(t4["math_code_benchmark_score"], 76.0)
        self.assertLess(t4["final_8t_validation_loss"], t1["final_8t_validation_loss"])


if __name__ == "__main__":
    unittest.main()

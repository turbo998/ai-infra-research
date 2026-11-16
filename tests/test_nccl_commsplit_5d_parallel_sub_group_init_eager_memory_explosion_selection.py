import unittest
from experiments.nccl_commsplit_5d_parallel_sub_group_init_eager_memory_explosion_selection.run import (
    evaluate_nccl_commsplit_tiers,
)


class TestNcclCommSplitSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_nccl_commsplit_tiers()
        self.assertAlmostEqual(res["naive_init_sec"], 1152.0, places=1)
        self.assertAlmostEqual(res["optimized_init_sec"], 18.5, places=1)
        self.assertAlmostEqual(res["init_speedup_x"], 62.3, places=1)
        self.assertAlmostEqual(res["hbm_saved_gb"], 12.95, places=2)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["nccl_hbm_per_gpu_gb"], 14.80)
        self.assertEqual(t4["nccl_hbm_per_gpu_gb"], 1.85)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()

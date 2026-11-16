import unittest
from experiments.ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection.run import (
    evaluate_ray_placement_tiers,
)


class TestRayPlacementSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_ray_placement_tiers()
        self.assertAlmostEqual(res["default_crash_rate_pct"], 87.5, places=1)
        self.assertAlmostEqual(res["spilled_pack_tpot_ms"], 118.5, places=1)
        self.assertAlmostEqual(res["strict_sorted_tpot_ms"], 14.2, places=1)
        self.assertAlmostEqual(res["tpot_speedup_x"], 8.3, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["actor_init_crash_rate_pct"], 87.5)
        self.assertEqual(t4["actor_init_crash_rate_pct"], 0.0)
        self.assertTrue(t4["cuda_ipc_weight_sync_supported"])
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()

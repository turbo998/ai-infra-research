import unittest
from experiments.rope_context_extension_pi_ntk_yarn_longrope_selection.run import (
    evaluate_rope_extension_tiers,
)


class TestRopeContextExtensionSelection(unittest.TestCase):
    def test_tiers_and_yarn_longrope_invariants(self) -> None:
        data = evaluate_rope_extension_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertAlmostEqual(data["workload"]["yarn_mscale"], 1.2773, places=3)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_linear_position_interpolation_pi"]
        t4 = tiers["tier4_longrope_piecewise_nonuniform_search"]
        self.assertLess(t1["short_4k_accuracy_drop_pts"], -5.0)
        self.assertEqual(t4["short_4k_accuracy_drop_pts"], 0.0)
        self.assertGreaterEqual(t4["long_128k_passkey_recall_pct"], 99.0)


if __name__ == "__main__":
    unittest.main()

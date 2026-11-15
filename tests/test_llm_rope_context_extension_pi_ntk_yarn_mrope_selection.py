import unittest
from experiments.llm_rope_context_extension_pi_ntk_yarn_mrope_selection.run import (
    evaluate_rope_context_extension_tiers,
)


class TestRopeContextExtensionSelection(unittest.TestCase):
    def test_rope_extension_tiers(self) -> None:
        data = evaluate_rope_context_extension_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_linear_position_interpolation_pi_uniform_downscale"]
        t4 = tiers["tier4_yarn_ntk_by_parts_plus_pre_scaled_cos_sin_temperature"]
        self.assertLess(t1["long_context_128k_ruler_pct"], 70.0)
        self.assertGreater(t4["long_context_128k_ruler_pct"], 91.5)
        self.assertEqual(t4["extra_kernel_latency_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()

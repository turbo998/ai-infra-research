import unittest
from experiments.pytorch_sdpa_silent_math_fallback_attn_mask_stride_alignment_selection.run import (
    evaluate_sdpa_fallback_tiers,
)


class TestPyTorchSdpaFallbackSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_sdpa_fallback_tiers()
        self.assertAlmostEqual(res["math_fallback_latency_ms"], 52.40, places=2)
        self.assertAlmostEqual(res["flash_strict_latency_ms"], 3.80, places=2)
        self.assertAlmostEqual(res["latency_speedup_x"], 13.8, places=1)
        self.assertAlmostEqual(res["hbm_reduction_x"], 128.0, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["attention_workspace_hbm_gb"], 32.00)
        self.assertEqual(t4["attention_workspace_hbm_gb"], 0.25)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()

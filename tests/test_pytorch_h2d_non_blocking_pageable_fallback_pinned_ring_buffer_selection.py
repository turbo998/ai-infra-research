import unittest
from experiments.pytorch_h2d_non_blocking_pageable_fallback_pinned_ring_buffer_selection.run import evaluate


class TestPyTorchH2DNonBlockingPageableFallbackPinnedRingBufferSelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_pageable_non_blocking_fallback"]
        t2 = res["tiers"]["tier2_per_step_dynamic_pin_memory"]
        t3 = res["tiers"]["tier3_single_pinned_buffer_unguarded"]
        t4 = res["tiers"]["tier4_pinned_double_ring_h2d_stream"]

        self.assertFalse(t1["truly_async_dma"])
        self.assertAlmostEqual(t1["cpu_exposed_stall_ms"], 8.40, places=2)
        self.assertAlmostEqual(t2["cpu_exposed_stall_ms"], 24.60, places=2)
        self.assertAlmostEqual(t3["overwrite_corruption_pct"], 14.2, places=1)
        self.assertAlmostEqual(t4["cpu_exposed_stall_ms"], 0.06, places=2)
        self.assertAlmostEqual(t4["overwrite_corruption_pct"], 0.0, places=1)
        self.assertAlmostEqual(t4["speedup_vs_tier1_x"], 140.0, places=1)


if __name__ == "__main__":
    unittest.main()

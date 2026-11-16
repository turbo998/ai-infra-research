import unittest
from experiments.llm_serving_zero_downtime_weight_hotswap_cumemunmap_kv_drain_selection.run import evaluate_zero_downtime_weight_hotswap


class TestZeroDowntimeWeightHotswap(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_zero_downtime_weight_hotswap()
        self.assertEqual(res["round"], 313)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_background_pinned_ram_preload_plus_inplace_dma_weight_refit"]["service_blackout_window_sec"],
            1.0,
        )
        self.assertFalse(
            tiers["tier4_background_pinned_ram_preload_plus_inplace_dma_weight_refit"]["cudagraph_recompilation_required"]
        )


if __name__ == "__main__":
    unittest.main()

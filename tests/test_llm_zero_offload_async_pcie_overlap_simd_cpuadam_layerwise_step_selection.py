import unittest
from experiments.llm_zero_offload_async_pcie_overlap_simd_cpuadam_layerwise_step_selection.run import evaluate_zero_offload_layerwise_cpuadam


class TestZeroOffloadLayerwiseCpuadam(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_zero_offload_layerwise_cpuadam()
        self.assertEqual(res["round"], 305)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_layerwise_bucketed_d2h_plus_overlapped_avx512_cpuadam_delayed_h2d"]["step_latency_sec"],
            3.6,
        )
        self.assertGreater(
            tiers["tier4_layerwise_bucketed_d2h_plus_overlapped_avx512_cpuadam_delayed_h2d"]["effective_mfu_pct"],
            58.0,
        )


if __name__ == "__main__":
    unittest.main()

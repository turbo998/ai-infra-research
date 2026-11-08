from __future__ import annotations

import unittest

from experiments.moe_quantization_parallelism_selection.run import (
    simulate_moe_quantization_parallelism,
)


class TestMoeQuantizationParallelismSelection(unittest.TestCase):
    def test_quantization_and_ep_invariants(self) -> None:
        res = simulate_moe_quantization_parallelism()
        bf16 = res["options"]["bf16_pure_tp16"]
        w4a16 = res["options"]["w4a16_awq_tp8"]
        fp8 = res["options"]["fp8_block_scaled_dp8_ep8"]
        fp4 = res["options"]["nvfp4_mxfp4_dp16_ep16"]
        eplb = res["eplb_skew_study"]

        self.assertEqual(bf16["weight_memory_gb"], 1342.0)
        self.assertEqual(bf16["cost_per_1m_tokens_usd"], 0.92)
        self.assertEqual(w4a16["cost_per_1m_tokens_usd"], 0.71)
        self.assertEqual(fp8["goodput_speedup_vs_bf16_tp16"], 2.65)
        self.assertEqual(fp8["cost_per_1m_tokens_usd"], 0.35)
        self.assertEqual(fp8["cost_reduction_pct_vs_bf16_tp16"], 62.0)
        self.assertEqual(fp4["goodput_speedup_vs_fp8"], 1.64)
        self.assertEqual(fp4["goodput_speedup_vs_bf16_tp16"], 4.34)
        self.assertEqual(fp4["cost_per_1m_tokens_usd"], 0.21)
        self.assertEqual(eplb["tpot_reduction_pct"], 37.1)
        self.assertEqual(eplb["goodput_gain_pct"], 53.1)
        self.assertFalse(res["provenance"]["independently_reproduced"])
        self.assertEqual(res["provenance"]["live_64xh100_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()

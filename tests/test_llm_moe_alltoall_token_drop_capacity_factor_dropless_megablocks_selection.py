import unittest
from experiments.llm_moe_alltoall_token_drop_capacity_factor_dropless_megablocks_selection.run import (
    evaluate_moe_dropless_grouped_gemm_tiers,
)


class TestMoeDroplessGroupedGemmSelection(unittest.TestCase):
    def test_moe_dropless_tiers(self) -> None:
        data = evaluate_moe_dropless_grouped_gemm_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_tight_capacity_factor_1_0_token_dropping_accuracy_loss"]
        t2 = tiers["tier2_high_capacity_factor_3_0_static_zero_padding_waste"]
        t4 = tiers["tier4_device_scheduled_dropless_ragged_grouped_gemm_deepgemm"]
        self.assertGreater(t1["tokens_dropped_pct"], 10.0)
        self.assertGreater(t2["padded_zero_flop_waste_pct"], 60.0)
        self.assertEqual(t4["tokens_dropped_pct"], 0.0)
        self.assertLess(t4["moe_layer_latency_ms"], 12.0)


if __name__ == "__main__":
    unittest.main()

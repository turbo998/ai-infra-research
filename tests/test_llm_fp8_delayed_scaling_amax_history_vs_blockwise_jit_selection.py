import unittest
from experiments.llm_fp8_delayed_scaling_amax_history_vs_blockwise_jit_selection.run import (
    evaluate_fp8_scaling_paradigm_tiers,
)


class TestFp8ScalingParadigmSelection(unittest.TestCase):
    def test_fp8_scaling_tiers(self) -> None:
        data = evaluate_fp8_scaling_paradigm_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_per_tensor_delayed_scaling_amax_history_1024"]
        t4 = tiers["tier4_online_1x128_tile_128x128_block_sram_fused_or_mxfp8"]
        self.assertLess(t1["quantization_snr_db"], 15.0)
        self.assertGreater(t4["quantization_snr_db"], 31.0)
        self.assertLessEqual(t4["val_loss_degradation_vs_bf16"], 0.002)
        self.assertGreater(t4["end_to_end_speedup_vs_bf16_x"], 1.45)


if __name__ == "__main__":
    unittest.main()

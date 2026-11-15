import unittest
from experiments.llm_qlora_nf4_double_quant_dequant_marlin_awq_selection.run import (
    evaluate_4bit_weight_quant_marlin_tiers,
)


class Test4BitWeightQuantMarlinSelection(unittest.TestCase):
    def test_4bit_marlin_tiers(self) -> None:
        data = evaluate_4bit_weight_quant_marlin_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_bitsandbytes_nf4_two_pass_hbm_dequant_materialization"]
        t4 = tiers["tier4_awq_gptq_with_marlin_or_machete_lop3_register_fused_kernel"]
        self.assertEqual(t1["hbm_traffic_bytes_per_param"], 4.5)
        self.assertLess(t1["speedup_vs_bf16_x"], 1.0)
        self.assertGreater(t4["speedup_vs_bf16_x"], 3.1)
        self.assertEqual(t4["decode_itl_ms"], 6.0)


if __name__ == "__main__":
    unittest.main()

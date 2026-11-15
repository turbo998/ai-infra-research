import unittest
from experiments.llm_multi_lora_serving_punica_sgmv_bgmv_rank_heterogeneity_selection.run import (
    evaluate_multi_lora_sgmv_slora_tiers,
)


class TestMultiLoraSgmvSloraSelection(unittest.TestCase):
    def test_multi_lora_tiers(self) -> None:
        data = evaluate_multi_lora_sgmv_slora_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t2 = tiers["tier2_unmerged_base_plus_python_for_loop_per_adapter_gemm"]
        t4 = tiers["tier4_punica_slora_sgmv_bgmv_plus_unified_paged_kv_and_lora_pool"]
        self.assertGreater(t2["effective_step_latency_ms"], 40.0)
        self.assertLess(t4["effective_step_latency_ms"], 7.0)
        self.assertGreater(t4["max_resident_adapters_in_hbm"], 300)


if __name__ == "__main__":
    unittest.main()

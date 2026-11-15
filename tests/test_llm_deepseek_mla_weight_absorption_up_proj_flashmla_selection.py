import unittest
from experiments.llm_deepseek_mla_weight_absorption_up_proj_flashmla_selection.run import (
    evaluate_deepseek_mla_duality_tiers,
)


class TestDeepseekMlaDualitySelection(unittest.TestCase):
    def test_deepseek_mla_tiers(self) -> None:
        data = evaluate_deepseek_mla_duality_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_explicit_up_projection_mha_for_both_prefill_and_decode"]
        t2 = tiers["tier2_weight_absorption_mqa_for_both_prefill_and_decode"]
        t4 = tiers["tier4_hybrid_prefill_upproj_fa3_plus_decode_weight_absorbed_flashmla"]
        self.assertGreater(t1["decode_step_latency_ms"], 80.0)
        self.assertGreater(t2["prefill_32k_latency_ms"], 600.0)
        self.assertEqual(t4["prefill_32k_latency_ms"], 210.0)
        self.assertEqual(t4["decode_step_latency_ms"], 6.2)


if __name__ == "__main__":
    unittest.main()

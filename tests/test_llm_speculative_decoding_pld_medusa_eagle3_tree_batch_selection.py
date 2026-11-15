import unittest
from experiments.llm_speculative_decoding_pld_medusa_eagle3_tree_batch_selection.run import (
    evaluate_speculative_decoding_batch_crossover_tiers,
)


class TestSpeculativeDecodingBatchCrossoverSelection(unittest.TestCase):
    def test_speculative_decoding_tiers(self) -> None:
        data = evaluate_speculative_decoding_batch_crossover_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_static_64node_tree_speculation_at_high_batch_negative_speedup"]
        t4 = tiers["tier4_eagle3_multilayer_feature_draft_plus_dynamic_batch_tree_throttling"]
        self.assertGreater(t1["high_batch_bs64_effective_tpot_ms"], 25.0)
        self.assertLess(t4["low_batch_bs1_effective_tpot_ms"], 3.2)
        self.assertLess(t4["high_batch_bs64_effective_tpot_ms"], 14.5)
        self.assertGreater(t4["mean_accepted_tokens_tau"], 4.5)


if __name__ == "__main__":
    unittest.main()

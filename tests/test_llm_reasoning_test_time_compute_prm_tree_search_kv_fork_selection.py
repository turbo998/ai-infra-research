import unittest
from experiments.llm_reasoning_test_time_compute_prm_tree_search_kv_fork_selection.run import (
    evaluate_test_time_compute_cascade_tree_tiers,
)


class TestTestTimeComputeCascadeTreeSelection(unittest.TestCase):
    def test_cascade_tree_tiers(self) -> None:
        data = evaluate_test_time_compute_cascade_tree_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_independent_n64_sequences_duplicate_prompt_kv_and_reads"]
        t2 = tiers["tier2_cow_paged_kv_sharing_with_naive_per_sequence_paged_attention"]
        t4 = tiers["tier4_multi_level_cascade_tree_attention_plus_prm_cow_fork_and_prune"]
        self.assertEqual(t1["prompt_kv_hbm_capacity_gb"], 128.0)
        self.assertEqual(t2["prompt_kv_hbm_capacity_gb"], 2.0)
        self.assertLess(t4["decode_step_latency_ms"], t2["decode_step_latency_ms"] / 8.0)
        self.assertGreater(t4["reasoning_benchmark_accuracy_pct"], 89.0)


if __name__ == "__main__":
    unittest.main()

import unittest
from experiments.llm_rl_rollout_length_skew_dynamic_microbatch_token_budget_selection.run import (
    evaluate_rl_dynamic_microbatch_lpt_tiers,
)


class TestRlDynamicMicrobatchLptSelection(unittest.TestCase):
    def test_rl_dynamic_bsz_tiers(self) -> None:
        data = evaluate_rl_dynamic_microbatch_lpt_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_fixed_sample_microbatch_right_padding_waste_and_oom"]
        t4 = tiers["tier4_global_lpt_karmarkar_karp_dp_balance_plus_dynamic_bsz"]
        self.assertGreater(t1["actor_ref_update_latency_sec"], 60.0)
        self.assertLess(t4["actor_ref_update_latency_sec"], 12.0)
        self.assertLess(t4["cross_dp_workload_skew_ratio"], 1.05)


if __name__ == "__main__":
    unittest.main()

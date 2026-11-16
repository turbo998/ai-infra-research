import unittest
from experiments.llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection.run import evaluate_checkpoint_tp_rng_philox


class TestCheckpointTpRngPhilox(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_checkpoint_tp_rng_philox()
        self.assertEqual(res["round"], 317)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertEqual(
            tiers["tier4_dual_tp_dp_rng_tracker_plus_graph_safe_device_philox_state"]["fwd_bwd_recompute_bitwise_match_pct"],
            100.0,
        )
        self.assertLess(
            tiers["tier4_dual_tp_dp_rng_tracker_plus_graph_safe_device_philox_state"]["rng_state_overhead_per_step_ms"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

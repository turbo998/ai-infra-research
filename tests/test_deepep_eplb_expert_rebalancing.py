import unittest

from experiments.deepep_eplb_expert_rebalancing.run import _check, run_simulation


class TestDeepEPEPLBExpertRebalancing(unittest.TestCase):
    def test_eplb_hierarchical_and_5tier_migration(self) -> None:
        res = run_simulation()
        _check(res)
        lb = res["load_balance_policies"]
        self.assertEqual(lb["hierarchical_eplb_288"]["distinct_groups_per_node"], [2, 2, 2, 2])
        self.assertLess(
            lb["hierarchical_eplb_288"]["max_over_mean_gpu_load_ratio"],
            lb["static_no_replication_256"]["max_over_mean_gpu_load_ratio"] * 0.55,
        )


if __name__ == "__main__":
    unittest.main()

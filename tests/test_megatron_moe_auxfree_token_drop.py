"""Unit tests for Megatron-Core MoE Aux-Loss-Free Routing (`expert_bias`), Quantile Balancing & Token Dropping."""

from __future__ import annotations

import subprocess
import sys
import unittest

from experiments.megatron_moe_auxfree_token_drop.run import (
    RESULTS_PATH,
    get_capacity,
    get_updated_expert_bias,
    qb_dual_update,
    run_moe_routing_experiment,
)


class TestMegatronMoEAuxFreeTokenDrop(unittest.TestCase):
    def test_get_capacity_and_expert_bias_sign_update(self) -> None:
        self.assertEqual(get_capacity(num_tokens=4096, num_experts=32, capacity_factor=1.25), 160)
        # Integer-exact sign update: total_tokens = 300, num_experts = 3 -> average = 100
        # expert 0: 150 (>100 -> -rate), expert 1: 100 (==100 -> 0), expert 2: 50 (<100 -> +rate)
        updated = get_updated_expert_bias([150, 100, 50], [0.0, 0.0, 0.0], 0.01)
        self.assertAlmostEqual(updated[0], -0.01)
        self.assertAlmostEqual(updated[1], 0.0)
        self.assertAlmostEqual(updated[2], 0.01)

    def test_qb_dual_update_reduces_imbalance(self) -> None:
        res = run_moe_routing_experiment()
        summary = res["summary"]
        self.assertGreater(summary["unbalanced_max_load_factor"], 7.0)
        self.assertLess(summary["expert_bias_max_load_factor"], 1.4)
        self.assertLess(summary["quantile_balancing_max_load_factor"], 1.1)

    def test_capacity_factor_token_drop_and_policy(self) -> None:
        res = run_moe_routing_experiment()
        summary = res["summary"]
        # Biased routing slashes CF=1.25 drop rate from 39.4% to < 0.5%
        self.assertGreater(summary["cf125_unbalanced_drop_rate_pct"], 35.0)
        self.assertLess(summary["cf125_biased_drop_rate_pct"], 0.5)
        # "probs" drop policy loses less probability mass than "position" drop policy
        self.assertLess(
            summary["cf125_probs_dropped_mass_pct"],
            summary["cf125_position_dropped_mass_pct"],
        )

    def test_cli_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, "experiments/megatron_moe_auxfree_token_drop/run.py", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertTrue(RESULTS_PATH.exists())


if __name__ == "__main__":
    unittest.main()

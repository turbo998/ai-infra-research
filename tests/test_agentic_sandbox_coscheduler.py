import unittest

from experiments.agentic_sandbox_coscheduler.run import (
    check_outputs,
    compute_breakeven_thresholds,
    generate_results,
    load_input,
)


class AgenticSandboxCoschedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_input()
        self.results = generate_results(self.config)

    def test_outputs_in_sync(self) -> None:
        self.assertTrue(check_outputs(self.results))

    def test_breakeven_thresholds_and_coscheduling_gain(self) -> None:
        thresholds = compute_breakeven_thresholds(self.config["cluster_parameters"])
        self.assertAlmostEqual(thresholds["kv_footprint_gib"], 4.0, places=2)
        self.assertAlmostEqual(thresholds["breakeven_gds_offload_sec"], 2.56, places=2)
        self.assertAlmostEqual(thresholds["breakeven_recompute_sec"], 8.0, places=2)

        summary = self.results["summary"]
        self.assertGreater(summary["tool_30s_rollout_gain_x"], 13.0)
        self.assertGreater(summary["tool_60s_rollout_gain_x"], 25.0)
        self.assertLess(
            summary["uffd_500vms_coscheduled_swap_sec"],
            summary["uffd_500vms_unthrottled_swap_sec"] * 0.15,
        )


if __name__ == "__main__":
    unittest.main()

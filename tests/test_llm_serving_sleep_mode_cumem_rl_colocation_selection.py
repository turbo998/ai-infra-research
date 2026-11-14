import unittest
from experiments.llm_serving_sleep_mode_cumem_rl_colocation_selection.run import (
    evaluate_sleep_mode_tiers,
)


class TestSleepModeCuMemSelection(unittest.TestCase):
    def test_sleep_mode_tiers(self) -> None:
        data = evaluate_sleep_mode_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_process_kill_and_cold_restart"]
        t3 = tiers["tier3_sleep_level1_cumem_cpu_pinned_offload"]
        t4 = tiers["tier4_sleep_level2_cumem_discard_plus_ipc_refit"]
        self.assertGreater(t1["wake_up_latency_s"], 60.0)
        self.assertTrue(t3["cudagraph_preserved"])
        self.assertLess(t3["wake_up_latency_s"], 1.0)
        self.assertLess(t4["wake_up_latency_s"], 0.3)


if __name__ == "__main__":
    unittest.main()

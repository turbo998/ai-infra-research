import unittest
from experiments.python_gc_pause_desync_straggler_gc_freeze_coordinated_collect_selection.run import (
    evaluate_python_gc_straggler_tiers,
)


class TestPythonGcStragglerSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_python_gc_straggler_tiers()
        self.assertAlmostEqual(res["unfrozen_gc_pause_ms"], 125.0, places=1)
        self.assertAlmostEqual(res["frozen_gc_pause_ms"], 3.2, places=1)
        self.assertAlmostEqual(res["gc_pause_speedup_x"], 39.1, places=1)
        self.assertAlmostEqual(res["cluster_throughput_gain_pct"], 51.9, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["straggler_step_waste_pct"], 34.2)
        self.assertEqual(t4["cyclic_memory_leak_gb"], 0.0)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()

import unittest
from experiments.llm_serving_stop_string_bpe_boundary_split_streaming_holdback_aho_corasick_selection.run import (
    evaluate_stop_string_tiers,
)


class TestStopStringSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_stop_string_tiers()
        self.assertAlmostEqual(res["token_id_missed_stop_pct"], 31.4, places=1)
        self.assertAlmostEqual(res["full_scan_cpu_ms"], 18.40, places=2)
        self.assertAlmostEqual(res["aho_corasick_cpu_ms"], 0.12, places=2)
        self.assertAlmostEqual(res["cpu_speedup_x"], 153.3, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["missed_stop_rate_pct"], 31.4)
        self.assertEqual(t4["missed_stop_rate_pct"], 0.0)
        self.assertEqual(t4["streaming_prefix_leak_pct"], 0.0)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()

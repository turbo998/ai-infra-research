import unittest
from experiments.ai_server_linux_thp_khugepaged_compaction_numa_balancing_jitter_selection.run import evaluate_linux_thp_numa_balancing_jitter


class TestLinuxThpNumaBalancingJitter(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_linux_thp_numa_balancing_jitter()
        self.assertEqual(res["round"], 311)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_numa_balancing_off_thp_madvise_plus_static_hugetlb_pinned_pool"]["cpu_pause_p99_jitter_ms"],
            1.0,
        )
        self.assertGreater(
            tiers["tier4_numa_balancing_off_thp_madvise_plus_static_hugetlb_pinned_pool"]["effective_cluster_mfu_pct"],
            58.0,
        )


if __name__ == "__main__":
    unittest.main()

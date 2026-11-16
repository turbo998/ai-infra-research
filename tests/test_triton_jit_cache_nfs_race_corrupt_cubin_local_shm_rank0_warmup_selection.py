import unittest
from experiments.triton_jit_cache_nfs_race_corrupt_cubin_local_shm_rank0_warmup_selection.run import (
    evaluate_triton_jit_cache_tiers,
)


class TestTritonJitCacheSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_triton_jit_cache_tiers()
        self.assertAlmostEqual(res["nfs_cold_startup_sec"], 840.0, places=1)
        self.assertAlmostEqual(res["rank0_barrier_cold_startup_sec"], 24.9, places=1)
        self.assertAlmostEqual(res["speedup_vs_nfs_x"], 33.7, places=1)
        self.assertAlmostEqual(res["speedup_vs_per_rank_x"], 6.7, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["corrupt_cubin_crash_rate_pct"], 38.5)
        self.assertEqual(t4["corrupt_cubin_crash_rate_pct"], 0.0)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()

import unittest
from experiments.deepep_engram_hybrid_symmetric.run import (
    _check,
    simulate_deepep_engram_hybrid_symmetric,
)


class TestDeepEpEngramHybridSymmetric(unittest.TestCase):
    def test_simulation_invariants(self) -> None:
        res = simulate_deepep_engram_hybrid_symmetric()
        _check(res)
        self.assertEqual(res["symmetric_vmm_memory"]["hbm_saved_cluster_gib"], 1024.0)
        self.assertGreater(res["engram_rdma_fetch"]["speedup_vs_cpu_proxy_x"], 20.0)


if __name__ == "__main__":
    unittest.main()

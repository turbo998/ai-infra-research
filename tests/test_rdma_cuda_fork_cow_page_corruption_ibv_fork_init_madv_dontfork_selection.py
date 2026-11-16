import unittest
from experiments.rdma_cuda_fork_cow_page_corruption_ibv_fork_init_madv_dontfork_selection.run import (
    evaluate_rdma_cuda_fork_tiers,
)


class TestRdmaCudaForkSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_rdma_cuda_fork_tiers()
        self.assertAlmostEqual(res["unprotected_cow_corruption_pct"], 22.5, places=1)
        self.assertEqual(res["tier4_cow_corruption_pct"], 0.0)
        self.assertAlmostEqual(res["spawn_speedup_x"], 805.6, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["silent_rdma_cow_corruption_pct"], 22.5)
        self.assertEqual(t4["silent_rdma_cow_corruption_pct"], 0.0)
        self.assertEqual(t4["subprocess_spawn_latency_ms"], 1.8)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()

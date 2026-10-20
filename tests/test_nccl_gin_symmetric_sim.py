import unittest
from experiments.nccl_gin_symmetric_sim.run import (
    _check,
    compute_lsa_gray_offsets,
    compute_rs_algo_hier_warps,
    simulate_nccl_gin_symmetric,
)


class TestNcclGinSymmetricSim(unittest.TestCase):
    def test_simulation_invariants(self) -> None:
        res = simulate_nccl_gin_symmetric()
        _check(res)
        self.assertEqual(
            res["allgather_rail_ring_lsa_stmc"]["gray_code_multipliers"],
            [0, 1, 3, 2, 4, 5, 7, 6],
        )
        self.assertGreater(
            res["rmsnorm_gin_fusion"]["gains"]["latency_speedup_x"], 2.0
        )

    def test_single_node_lsa1_warp_partition(self) -> None:
        part = compute_rs_algo_hier_warps(cta_threads=512, lsa_nranks=1, rail_nranks=8, multimem=False)
        self.assertEqual(part["stage0_work_warps"], 0)
        self.assertEqual(part["stage1_work_warps"], 14)

    def test_gray_code_bijection(self) -> None:
        rows = compute_lsa_gray_offsets(8, 262144, 4 * 1024 * 1024, 65536)
        self.assertEqual(len({r["start_offset_bytes"] for r in rows}), 8)


if __name__ == "__main__":
    unittest.main()

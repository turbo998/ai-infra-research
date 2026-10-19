import unittest
from experiments.mooncake_cx8_data_direct_rdma.run import (
    _check,
    simulate_mooncake_cx8_data_direct_rdma,
)


class TestMooncakeCx8DataDirectRdma(unittest.TestCase):
    def test_simulation_invariants(self) -> None:
        res = simulate_mooncake_cx8_data_direct_rdma()
        _check(res)
        self.assertGreater(res["gains"]["submit_cpu_speedup_x"], 34.0)
        self.assertGreater(res["gains"]["total_speedup_vs_round_robin_x"], 4.8)


if __name__ == "__main__":
    unittest.main()

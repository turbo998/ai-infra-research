import unittest
from experiments.megatron_fsdp_nccl_ub_hfsdp.run import (
    _check,
    simulate_megatron_fsdp_nccl_ub_hfsdp,
)


class TestMegatronFsdpNcclUbHfsdp(unittest.TestCase):
    def test_simulation_invariants(self) -> None:
        res = simulate_megatron_fsdp_nccl_ub_hfsdp()
        _check(res)
        self.assertTrue(res["strategies"]["hsdp_outer_no_shard"]["oom_on_192gb_b200"])
        self.assertGreater(res["key_gains"]["step_speedup_vs_full_fsdp_x"], 1.34)


if __name__ == "__main__":
    unittest.main()

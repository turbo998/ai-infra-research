import unittest
from experiments.deepgemm_mega_moe_mhc_sim.run import (
    _check,
    get_block_config_for_mega_moe,
    get_mhc_num_splits,
    simulate_deepgemm_mega_moe_mhc,
)


class TestDeepGemmMegaMoeMhcSim(unittest.TestCase):
    def test_simulation_invariants(self) -> None:
        res = simulate_deepgemm_mega_moe_mhc()
        _check(res)
        self.assertEqual(
            res["asymmetric_store_block_240_comparison"]["smem_cd_saved_bytes"],
            14336,
        )
        self.assertGreater(res["mega_mhc_fusion"]["mhc_speedup_x"], 1.9)

    def test_few_local_experts_two_block_split(self) -> None:
        # When num_experts / num_ranks < 14 and covered_tokens > 192, uses 2 blocks of 128 instead of 1 block of 240
        cfg = get_block_config_for_mega_moe(
            num_ranks=8,
            num_experts=64,
            num_max_tokens_per_rank=256,
            num_topk=8,
            num_tokens=200,
            mma_kind="MXFP8FP4",
        )
        self.assertEqual(cfg["block_m"], 128)

    def test_mhc_split_k_bounds(self) -> None:
        splits = get_mhc_num_splits(num_m_blocks=64, hidden=7168, num_sms=148)
        self.assertEqual(splits, 14)


if __name__ == "__main__":
    unittest.main()

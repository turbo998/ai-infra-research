import unittest

from experiments.torchtitan_flexshard_hybridep_sim.run import (
    BlockShardModel,
    num_permuted_tokens_for_non_blocking,
    run_simulation,
    simulate_kimi_wkva_block_sharding,
)


class TestTorchTitanFlexShardHybridEPSim(unittest.TestCase):
    def test_block_shard_heterogeneous_periods(self) -> None:
        bs = BlockShardModel(0, (512, 64))
        self.assertEqual(bs.num_blocks(4608), 16)
        self.assertEqual(bs.block_start(0), 0)
        self.assertEqual(bs.block_start(1), 512)
        self.assertEqual(bs.block_start(2), 576)
        self.assertEqual(bs.block_start(3), 1088)

        with self.assertRaises(ValueError):
            bs.num_blocks(4096)

    def test_kimi_wkva_and_muon_overlap(self) -> None:
        kimi = simulate_kimi_wkva_block_sharding()
        self.assertEqual(kimi["total_blocks"], 16)
        self.assertGreater(kimi["naive_6rank_boundary_violations"], 0)

        sim = run_simulation()
        self.assertGreater(sim["bucketed_muon_overlap"]["step_speedup_ratio"], 1.5)

    def test_hybridep_non_blocking_alignment(self) -> None:
        n = num_permuted_tokens_for_non_blocking(
            num_tokens=4096,
            ep_size=72,
            num_local_experts=4,
            top_k=8,
            moe_expert_capacity_factor=0.25,
            pad_multiple=128,
        )
        self.assertEqual(n % 128, 0)
        self.assertEqual(n, 294912)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from experiments.long_context_attention.run import load_input, run_experiment


class LongContextAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_input()
        self.results = run_experiment(self.spec)

    def test_kv_format_exact_bytes_and_1m_footprint(self) -> None:
        by_id = {row["id"]: row for row in self.results["kv_format_comparison"]}
        self.assertEqual(by_id["gqa8_bf16_dense"]["bytes_per_token_per_layer"], 4096)
        self.assertEqual(by_id["deepseek_v3_mla_bf16_dense"]["bytes_per_token_per_layer"], 1152)
        self.assertEqual(by_id["deepseek_v3_2_mla_fp8_dsa"]["bytes_per_token_per_layer"], 656)
        self.assertEqual(by_id["deepseek_v4_1_mla_fp8_dsa"]["bytes_per_token_per_layer"], 528)
        self.assertEqual(by_id["deepseek_v4_1_mla_fp4_dsa"]["bytes_per_token_per_layer"], 288)
        self.assertAlmostEqual(by_id["gqa8_bf16_dense"]["kv_cache_gib_per_1m_req"], 244.0, places=4)
        self.assertAlmostEqual(by_id["deepseek_v4_1_mla_fp8_dsa"]["kv_cache_gib_per_1m_req"], 31.4531, places=4)
        self.assertAlmostEqual(by_id["deepseek_v4_1_mla_fp4_dsa"]["kv_cache_gib_per_1m_req"], 17.1562, places=4)

    def test_hopper_dsm_crossover_cycles(self) -> None:
        dsm = self.results["hopper_dsm_crossover"]
        self.assertAlmostEqual(dsm["mma_cycles_per_kv_token"], 34.0, places=4)
        self.assertAlmostEqual(dsm["single_cta_dequant_cycles_per_kv_token"], 50.0, places=4)
        self.assertAlmostEqual(dsm["crossover_2cta_dequant_cycles_per_kv_token"], 25.0, places=4)
        self.assertEqual(dsm["single_cta_bound"], "dequantization-bound")
        self.assertEqual(dsm["crossover_2cta_bound"], "mma-bound")

    def test_context_parallelism_mla_head_constraint(self) -> None:
        cp_by_id = {row["id"]: row for row in self.results["context_parallelism_comparison"]}
        self.assertTrue(cp_by_id["ring_cp8_mla_mqa1"]["zero_kv_replication_within_heads"])
        self.assertFalse(cp_by_id["ulysses_cp8_mla_mqa1"]["zero_kv_replication_within_heads"])
        self.assertLess(
            cp_by_id["ring_cp8_mla_mqa1"]["comm_mib_per_layer_per_gpu"],
            cp_by_id["ulysses_cp8_mla_mqa1"]["comm_mib_per_layer_per_gpu"],
        )

    def test_hybrid_mamba_int8_checkpoint_capacity(self) -> None:
        mamba = self.results["hybrid_mamba_cache"]
        self.assertGreater(mamba["int8_slot_capacity_multiplier"], 1.8)
        self.assertEqual(mamba["max_cached_nodes_in_budget_bf16"], 351)
        self.assertEqual(mamba["max_cached_nodes_in_budget_int8"], 661)


if __name__ == "__main__":
    unittest.main()

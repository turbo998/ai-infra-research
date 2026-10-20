import unittest
from experiments.vllm_v1_sparse_mla_hisparse_sim.run import (
    _check,
    hisparse_residency_smem_bytes,
    nvfp4_ds_mla_scale_byte_offset,
    plan_flashinfer_checkpoints,
    simulate_vllm_v1_sparse_mla_hisparse,
)


class TestVllmV1SparseMlaHiSparseSim(unittest.TestCase):
    def test_simulation_invariants(self) -> None:
        res = simulate_vllm_v1_sparse_mla_hisparse()
        _check(res)
        self.assertEqual(
            res["sparse_mla_kv_formats"]["sm100_nvfp4_ds_mla"]["bytes_per_token"],
            352,
        )
        self.assertTrue(res["nvfp4_scale_permutation"]["is_contiguous_8byte_load"])

    def test_nvfp4_scale_permutation_bijection(self) -> None:
        offsets = [nvfp4_ds_mla_scale_byte_offset(s) for s in range(32)]
        self.assertEqual(sorted(offsets), list(range(32)))

    def test_hisparse_smem_and_gdn_checkpoints(self) -> None:
        smem = hisparse_residency_smem_bytes(max_query_len=1, model_top_k=2048)
        self.assertEqual(smem["required_smem_bytes"], 26148)
        plan = plan_flashinfer_checkpoints([0, 64, 128], [128, 192])
        self.assertEqual(plan["every_n_tokens"], 64)
        self.assertEqual(plan["num_checkpoints"], 5)


if __name__ == "__main__":
    unittest.main()

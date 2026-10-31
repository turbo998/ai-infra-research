import unittest
from experiments.flashinfer_qwen35_gdn_cake_softmax_fa_sm100_non32_phase170.run import run_model as run_gdn
from experiments.vllm_rust_block_hash_rocm_dsv41_trtllm_nemotron_phase170.run import run_model as run_vllm
from experiments.nccl_efa_dp_direct110_hipfile_host_nixl_tsan_phase170.run import run_model as run_nccl


class TestPhase170Experiments(unittest.TestCase):
    def test_flashinfer_qwen35_gdn_cake_softmax_fa_sm100_non32(self):
        res = run_gdn()
        self.assertEqual(res["fused_gdn_kernels"], 1)
        self.assertGreaterEqual(res["gdn_speedup_x"], 2.2)
        self.assertEqual(res["balanced_cta_chunk_elems"], 19008)
        self.assertEqual(res["padded_tmem_hdims"], [96, 128])
        self.assertEqual(res["jit_callable_default_collisions"], 0)

    def test_vllm_rust_block_hash_rocm_dsv41_trtllm_nemotron(self):
        res = run_vllm()
        self.assertGreaterEqual(res["rust_hash_speedup_x"], 11.0)
        self.assertEqual(res["flydsl_post_warmup_recompiles"], 0)
        self.assertEqual(res["gfx95_fp8_grid_max_bf16_ulp"], 1)
        self.assertTrue(res["harmony_final_channel_opened"])

    def test_nccl_efa_dp_direct110_hipfile_host_nixl_tsan(self):
        res = run_nccl()
        self.assertEqual(res["efa_gda_backend_version"], 2)
        self.assertEqual(res["efa_dp_direct_qp_bytes"], 144)
        self.assertEqual(res["efa_inline_rdma_write_bytes"], 128)
        self.assertEqual(res["host_backend_dram_seg_errors"], 0)
        self.assertEqual(res["tsan_etcd_invalidation_timeouts"], 0)


if __name__ == "__main__":
    unittest.main()

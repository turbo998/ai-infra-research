import unittest
from experiments.flashinfer_grouped_fp8_silu_quant_lmhead_loss_dsv4_tmem_deepseek_route_cudnn_kv_tilelang_phase98 import run as run_a
from experiments.sglang_bcg_nemotron_holder_lse_npu_bitmask_vllm_triton_av_wna16_jamba_mooncake_trtllm_phase98 import run as run_b
from experiments.lmcache_musa_mlu_nixl_libfabric_notif_megatron_mixtral_te_onnx_titan_softcap_verl_mxfp8_mooncake_dynamo_phase98 import run as run_c


class TestPhase98Experiments(unittest.TestCase):
    def test_track_a_flashinfer_and_tilelang_phase98(self):
        res = run_a.run_all()
        self.assertTrue(res["all_invariants_passed"])
        self.assertEqual(res["grouped_fp8_silu_quant_and_rebind"]["cuda_graph_speedup_max"], 1.154)
        self.assertEqual(res["lm_head_loss_dsv4_tmem_and_cudnn"]["cudnn_effective_s_kv"], 48)
        self.assertEqual(res["tilelang_ascend_and_cuda_math"]["strided_slice_new_offset"], 388)

    def test_track_b_sglang_vllm_trtllm_phase98(self):
        res = run_b.run_all()
        self.assertTrue(res["all_invariants_passed"])
        self.assertEqual(res["vllm_triton_av_wna16_jamba_mooncake"]["mooncake_descriptors_after"], 1)
        self.assertEqual(res["vllm_triton_av_wna16_jamba_mooncake"]["triton_av_draft_budget"], 4)

    def test_track_c_lmcache_nixl_megatron_te_titan_verl_mooncake_dynamo_phase98(self):
        res = run_c.run_all()
        self.assertTrue(res["all_invariants_passed"])
        self.assertEqual(res["lmcache_nixl_and_mooncake_rdma"]["nixl_lookup_reduction_x"], 256)
        self.assertEqual(res["lmcache_nixl_and_mooncake_rdma"]["mooncake_post_send_speedup_x"], 3.316)


if __name__ == "__main__":
    unittest.main()

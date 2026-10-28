from __future__ import annotations

import unittest

from experiments.flashinfer_cake_mla_sampling_vsa_attnres_gdn_tilelang_triton_phase128.run import (
    run_experiment as run_track_a,
)
from experiments.sglang_routed_kv_dsv4_npu_vllm_deepselect_mxfp8_nixl_phase128.run import (
    run_experiment as run_track_b,
)
from experiments.trtllm_helix_gdn_megatron_gdp_pytorch_inductor_mooncake_dynamo_phase128.run import (
    run_experiment as run_track_c,
)


class TestPhase128Experiments(unittest.TestCase):
    def test_track_a_flashinfer_tilelang_triton(self) -> None:
        res = run_track_a()
        self.assertEqual(res["phase"], 128)
        self.assertAlmostEqual(res["nvfp4_mla_max_speedup_x"], 1.029, places=3)
        self.assertEqual(res["sampling_round17"]["total_builds_after"], 150)
        self.assertAlmostEqual(res["sampling_round17"]["rtx_pro_6000_max_speedup_x"], 1.413, places=3)
        self.assertEqual(res["vsa_sm90_pin"]["main_loop_inst_delta"], -42)
        self.assertEqual(res["kimi_k3_attn_res_park"]["tmem_cols_per_stream_after"], 512)
        self.assertEqual(res["gdn_train_kcp"]["prepare_sync_calls_after"], 0)
        self.assertEqual(res["gdn_train_kcp"]["bwd_peak_saved_mib"], 2324)
        self.assertAlmostEqual(res["moe_ep_and_w4a16"]["dispatch_payload_reduction_pct"], 48.44, places=2)

    def test_track_b_sglang_vllm(self) -> None:
        res = run_track_b()
        self.assertEqual(res["phase"], 128)
        self.assertEqual(res["sglang_dsv4_npu_nosync"]["host_sync_ops_per_decode_step_after"], 0)
        self.assertAlmostEqual(res["vllm_dsa_deepselect_rows"][0]["speedup_x"], 4.9, places=1)
        self.assertAlmostEqual(res["vllm_dsa_e2e_gb200"]["upcast_plus_topk_speedup_x"], 6.912, places=3)
        self.assertAlmostEqual(res["vllm_rocm_minimax_mxfp8"]["norm_plus_quant_max_speedup_x"], 2.06, places=2)
        self.assertEqual(res["vllm_nixl_transfer_mode"]["resolved_push_class"], "NixlPushConnector")

    def test_track_c_trtllm_megatron_pytorch_mooncake_dynamo(self) -> None:
        res = run_track_c()
        self.assertEqual(res["phase"], 128)
        self.assertAlmostEqual(res["trtllm_helix_and_gdn"]["tp64_alltoall_payload_reduction_x"], 2.0, places=2)
        self.assertEqual(res["megatron_gdp_and_batch"]["bf16_sigmoid_grad_at_8_5"], 0.0)
        self.assertGreater(res["megatron_gdp_and_batch"]["fp32_sigmoid_grad_at_8_5"], 0.0002)
        self.assertFalse(res["megatron_gdp_and_batch"]["legacy_detected_transition"])
        self.assertTrue(res["megatron_gdp_and_batch"]["fixed_detected_transition"])
        self.assertTrue(res["pytorch_inductor_and_cuda"]["pointwise_cat_fused_after"])
        self.assertEqual(res["mooncake_and_dynamo"]["mooncake_shared_dso_singleton_count_after"], 1)


if __name__ == "__main__":
    unittest.main()

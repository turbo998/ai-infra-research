import unittest

from experiments.flashinfer_qsa_nvfp4_minimax_mxfp8_fp8_attn_mla_gdp_gated_act_tilelang_phase96.run import (
    run_all as run_track1,
)
from experiments.lmcache_lookup_state_nixl_s3_rdma_parts_megatron_mimo_te_titan_verl_mooncake_dynamo_phase96.run import (
    run_all as run_track3,
)
from experiments.sglang_beam_dsv4_bridges_kimi_tgv_dsv41_table_wan_ulysses_vllm_lora_mrv2_trtllm_phase96.run import (
    run_all as run_track2,
)


class TestPhase96Experiments(unittest.TestCase):
    def test_track1_flashinfer_and_tilelang_phase96(self) -> None:
        res = run_track1()
        self.assertEqual(res["status"], "verified")
        self.assertEqual(res["qsa_nvfp4_and_fp8_attn"]["direct_epilogue_smem_roundtrips"], 0)
        self.assertEqual(res["grouped_mxfp8_and_gated_act"]["total_packed_programs"], 18)
        self.assertEqual(res["gdp_tail_wave_and_tilelang"]["selected_gdp_t1_program"], "one_row_per_thread_l8r4")
        self.assertFalse(res["gdp_tail_wave_and_tilelang"]["sm8x_invalid_fp16_48x128"])

    def test_track2_sglang_vllm_trtllm_phase96(self) -> None:
        res = run_track2()
        self.assertEqual(res["status"], "verified")
        self.assertEqual(res["dsv4_prefill_bridges_and_kimi_tgv"]["stale_graph_pointer_reads"], 0)
        self.assertEqual(res["wan_direct_ulysses_and_dsv41_cp"]["intermediate_full_qk_writes"], 0)
        self.assertEqual(res["vllm_mrv2_lora_and_decode_bench"]["released_after_encoder_cpu_mb"], 0.0)
        self.assertTrue(res["vllm_mrv2_lora_and_decode_bench"]["all_groups_loaded"])

    def test_track3_lmcache_nixl_mooncake_verl_dynamo_phase96(self) -> None:
        res = run_track3()
        self.assertEqual(res["status"], "verified")
        self.assertEqual(res["lmcache_and_nixl_rdma"]["num_rdma_parts"], 3)
        self.assertEqual(res["lmcache_and_nixl_rdma"]["status_after_rejected_release"], "NIXL_IN_PROG")
        self.assertEqual(res["mooncake_evictor_and_dynamo_zmq"]["thread_reduction_ratio"], 32.0)
        self.assertTrue(res["verl_sampler_and_titan_mimo"]["impossible_mass_rejected"])


if __name__ == "__main__":
    unittest.main()

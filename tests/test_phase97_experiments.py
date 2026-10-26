import unittest

from experiments.flashinfer_megamoe_fence_minimax_sm120_grouped_fp8_mm_fp4_ar_union_mla_tilelang_phase97.run import (
    run_all as run_track_a,
)
from experiments.lmcache_p2p_dram_put_ref_nixl_ucx_cuobj_megatron_gdn_te_mxfp8_titan_verl_mooncake_dynamo_phase97.run import (
    run_all as run_track_c,
)
from experiments.sglang_qwen4_qsa_ple_mooncake_tag_quark_mxfp4_ascend_vllm_eagle_engram_octave_trtllm_phase97.run import (
    run_all as run_track_b,
)


class TestPhase97Experiments(unittest.TestCase):
    def test_track_a_flashinfer_and_tilelang(self) -> None:
        res = run_track_a()
        self.assertTrue(res["all_invariants_passed"])
        self.assertTrue(res["megamoe_ep16_alias_fence"]["alias_ordered_after_fix"])
        self.assertEqual(res["minimax_h3_and_grouped_fp8"]["fp8_m33472_speedup_x"], 1.0208)
        self.assertEqual(res["mm_fp4_ar_union_mla_splitk"]["mm_fp4_stages_after"], 6)
        self.assertEqual(res["tilelang_compiler_invariants"]["bf16x2_decoded_pair"], [-1.5, 2.25])

    def test_track_b_sglang_vllm_trtllm(self) -> None:
        res = run_track_b()
        self.assertTrue(res["all_invariants_passed"])
        self.assertEqual(res["sglang_serving"]["is_cpu_uname_calls_when_disabled"], 0)
        self.assertEqual(res["vllm_and_trtllm"]["engram_comm_reduction_x"], 8.0)
        self.assertEqual(res["vllm_and_trtllm"]["rocm_octave_compression_vs_fp16_x"], 5.3333)

    def test_track_c_lmcache_nixl_megatron_te_titan_verl_mooncake_dynamo(self) -> None:
        res = run_track_c()
        self.assertTrue(res["all_invariants_passed"])
        self.assertEqual(res["lmcache"]["p2p_admitted_tiers"], ["DRAM:LOCAL"])
        self.assertEqual(res["nixl"]["cuobj_driver_get_descriptor_calls"], 1)
        self.assertEqual(res["training_and_comm"]["torchtitan_shared_graph_train_eval_gib"], 61.530)
        self.assertEqual(res["training_and_comm"]["dynamo_task_reduction_x"], 1024.0)


if __name__ == "__main__":
    unittest.main()

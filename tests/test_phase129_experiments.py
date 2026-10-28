import unittest

from experiments.flashinfer_moe_cache_key_sm110_aot_tilelang_restrict_triton_phase129 import run as exp1
from experiments.sglang_mega_runtime_eagle_4096_engram_vllm_sm120_mla_sleep_phase129 import run as exp2
from experiments.pytorch_rccl_symm_tokenswitch_tma_mooncake_efa_lmcache_dynamo_phase129 import run as exp3


class TestPhase129Experiments(unittest.TestCase):
    def test_track1_flashinfer_tilelang_triton_phase129(self) -> None:
        s = exp1.build_summary()
        exp1.verify(s)
        self.assertEqual(s["flashinfer_cute_dsl_moe_cache_key"]["fixed_unique_keys_across_ep"], 1)
        self.assertEqual(s["flashinfer_sm110_aot_provider"]["fixed_sm110_aot_shipped"], 8)
        self.assertEqual(s["triton_fpsan_wmma_thread_region_gluon"]["fixed_loop_barriers"], 0)

    def test_track2_sglang_vllm_phase129(self) -> None:
        s = exp2.build_summary()
        exp2.verify(s)
        self.assertEqual(s["sglang_eagle_4096_rotation"]["trip_reduction_x"], 512.0)
        self.assertEqual(s["vllm_sm120_mla_sleep_and_pp_layout"]["compact_nope_entry_bytes"], 528)
        self.assertEqual(s["vllm_sm120_mla_sleep_and_pp_layout"]["pp_heterogeneous_layout_assertion_errors"], 0)

    def test_track3_pytorch_megatron_mooncake_lmcache_dynamo_phase129(self) -> None:
        s = exp3.build_summary()
        exp3.verify(s)
        self.assertEqual(s["mooncake_efa_hugepage_registration"]["max_4kb_reg_per_nic_gib"], 88)
        self.assertGreater(s["lmcache_concurrent_tp_transfers"]["lmcache_tp_bench_speedup_x"], 7.5)
        self.assertEqual(s["dynamo_dcgm_and_session_affinity"]["fixed_leaked_sockets_per_hour"], 0)


if __name__ == "__main__":
    unittest.main()

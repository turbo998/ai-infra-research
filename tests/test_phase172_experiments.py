import unittest
from experiments.flashinfer_moe_ckpt_b300_rope_triton_gsan_mbarrier_phase172.run import run_model as run_fi
from experiments.vllm_kv_reset_reentrant_nemotron_sglang_reqkvinfo_phase172.run import run_model as run_vllm
from experiments.rccl_dda_epoch_zero_rocjitsu_f32_nixl_infinia25_phase172.run import run_model as run_rccl


class TestPhase172Experiments(unittest.TestCase):
    def test_flashinfer_moe_ckpt_b300_rope_triton_gsan_mbarrier(self):
        res = run_fi()
        self.assertEqual(res["moe_ckpt_early_stop_fwd_recomputes"], 0)
        self.assertTrue(res["trtllm_gen_h512_nvfp4_kv_fallback_enabled"])
        self.assertEqual(res["deepgemm_runtime_cache_miss_after_load"], 0)
        self.assertEqual(res["gsan_mbarrier_table_oob_faults"], 0)

    def test_vllm_kv_reset_reentrant_nemotron_sglang_reqkvinfo(self):
        res = run_vllm()
        self.assertEqual(res["reentrant_reset_cache_deadlocks"], 0)
        self.assertEqual(res["nemotron_vl_norm_speedup_x"], 12.0)
        self.assertTrue(res["derender_prompt_logprobs_preserved"])
        self.assertEqual(res["deepep_supported_hidden_dim_3072"], 3072)

    def test_rccl_dda_epoch_zero_rocjitsu_f32_nixl_infinia25(self):
        res = run_rccl()
        self.assertEqual(res["uninitialized_dda_epoch_ll_races"], 0)
        self.assertEqual(res["rocjitsu_zero_sel_f32_store_corruptions"], 0)
        self.assertEqual(res["ddn_infinia_sdk_version"], "2.5")


if __name__ == "__main__":
    unittest.main()

import unittest
from experiments.flashinfer_kimik3_router_sf_underflow_fa_sm100_tmem_phase169.run import run_suite as run_s1
from experiments.deepep_cumem_shared_sglang_efa151_trtllm_mpi_drain_phase169.run import run_suite as run_s2
from experiments.nccl_tls_psk_efa_bit_rocjitsu_encoding_gfx1250_phase169.run import run_suite as run_s3


class TestPhase169Experiments(unittest.TestCase):
    def test_suite1_flashinfer_kimik3_sf_fa_sm100_tmem(self) -> None:
        res = run_s1()
        self.assertEqual(res["phase"], 169)
        self.assertEqual(res["flashinfer_kimik3_replayssm_sf_cudnn"]["fp8_sf_underflow_sites_guarded"], 5)
        self.assertEqual(res["fa_sm100_hdim64_bwd_dedicated_tmem"]["tmem_column_utilization_pct"], 100.0)
        self.assertEqual(res["fa_empty_offset_window_guard"]["fixed_empty_window_active_tokens"], 0)

    def test_suite2_deepep_cumem_trtllm_mpi_sglang_vllm(self) -> None:
        res = run_s2()
        self.assertEqual(res["phase"], 169)
        self.assertEqual(res["deepep_cumem_cross_process_shm"]["cumem_dynamic_remap_stalls"], 0)
        self.assertEqual(res["trtllm_mpi_pool_drain_before_fork"]["drained_child_spawn_deadlocks"], 0)
        self.assertEqual(res["sglang_efa_and_vllm_rocm_aiter"]["rocm_ci_image_reduction_ratio"], 4.0)

    def test_suite3_nccl_psk_rocjitsu_gfx1250_nixl_mooncake(self) -> None:
        res = run_s3()
        self.assertEqual(res["phase"], 169)
        self.assertEqual(res["nccl_tls_psk_and_efa_phase_bit"]["unauthenticated_non_psk_accepted_after_fix"], 0)
        self.assertEqual(res["rocjitsu_encoding_and_gfx1250_trap"]["trace_memory_footprint_reduction_ratio"], 8.0)
        self.assertEqual(res["nixl_log_demotion_and_mooncake_jit"]["mooncake_moe_cold_jit_timeout_sec"], 360)


if __name__ == "__main__":
    unittest.main()

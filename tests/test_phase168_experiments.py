import unittest
from experiments.flashinfer_frost_fp4_rubin_megamoe_fa_sm90_bwd_phase168.run import run_suite as run_s1
from experiments.sglang_swa_async_hicache_mlock_deepep_internode_graph_phase168.run import run_suite as run_s2
from experiments.nccl_efa_inline_put_cft_nvls_rocm_clock64_phase168.run import run_suite as run_s3


class TestPhase168Experiments(unittest.TestCase):
    def test_suite1_flashinfer_frost_rubin_fa_sm90(self) -> None:
        res = run_s1()
        self.assertEqual(res["phase"], 168)
        self.assertEqual(res["flashinfer_frost_rubin_tcgen05"]["invalid_route_nan_contaminations"], 0)
        self.assertEqual(res["fa_sm90_bwd_dkv_swapab_hdim192"]["wgmma_m64_remainder_after_fix"], 0)
        self.assertEqual(res["fa_causal_sliding_window_zero_bound"]["scheduler_kv_overplan_reduction_ratio"], 33.0)

    def test_suite2_sglang_swa_hicache_deepep_vllm(self) -> None:
        res = run_s2()
        self.assertEqual(res["phase"], 168)
        self.assertEqual(res["sglang_swa_hicache_ue8m0"]["async_host_syncs_per_step"], 0)
        self.assertEqual(res["deepep_internode_cudagraph_and_trtllm_prefetch"]["per_step_dispatch_cpu_overhead_reduction_ratio"], 7.4)
        self.assertEqual(res["vllm_rust_tool_call_delta_accounting"]["fixed_counted_tokens"], 300)

    def test_suite3_nccl_efa_inline_rocm_clock64(self) -> None:
        res = run_s3()
        self.assertEqual(res["phase"], 168)
        self.assertEqual(res["nccl_efa_gda_inline_put_and_cft"]["pcie_transaction_reduction_ratio"], 2.0)
        self.assertEqual(res["rocm_hip_tests_clock64_removal"]["emulation_timeout_failures_after_fix"], 0)
        self.assertTrue(res["nixl_mooncake_lmcache_governance"]["nixl_prometheus_mp_documented"])


if __name__ == "__main__":
    unittest.main()

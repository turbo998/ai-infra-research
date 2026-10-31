import unittest
from experiments.flashinfer_sm120_mqa_cake_lmhead_fa_sm100_sink_phase167.run import run_suite as run_s1
from experiments.sglang_dsv41_sm90_disagg_stop_vllm_fused_topk_phase167.run import run_suite as run_s2
from experiments.nccl_efa_gda_flush_rocjitsu_vmem_lmcache_inproc_phase167.run import run_suite as run_s3


class TestPhase167Experiments(unittest.TestCase):
    def test_suite1_flashinfer_sm120_cake_fa_sm100(self) -> None:
        res = run_s1()
        self.assertEqual(res["phase"], 167)
        self.assertEqual(res["sm120_paged_fp8_mqa_logits"]["routed_rtx5090_h64_nextn56_shapes"], 9)
        self.assertEqual(res["cake_lm_head_loss_sm100a"]["expect_tx_order_hazards_after_fix"], 0)
        self.assertEqual(res["fa_sm100_softmax_spill_and_mla_sink"]["optimized_spill_bytes_per_thread"], 0)

    def test_suite2_sglang_dsv41_disagg_vllm_topk(self) -> None:
        res = run_s2()
        self.assertEqual(res["phase"], 167)
        self.assertEqual(res["sglang_dsv41_sm90_tile_balance"]["tail_step_speedup_ratio"], 64.0)
        self.assertEqual(res["sglang_disagg_and_mamba_hicache"]["mamba_hicache_gpu_slots_required"], 0)
        self.assertEqual(res["vllm_fused_scoring_topk_prefill"]["memory_traffic_reduction_ratio"], 64.0)

    def test_suite3_nccl_efa_rocjitsu_mooncake_lmcache(self) -> None:
        res = run_s3()
        self.assertEqual(res["phase"], 167)
        self.assertEqual(res["nccl_efa_gda_flush_wait"]["unattributed_peer_completions"], 0)
        self.assertEqual(res["rocjitsu_vmem_gds_exp_accounting"]["fixed_false_positive_race_reports"], 0)
        self.assertEqual(res["mooncake_nixl_lmcache_fabric"]["nixl_rocm_wheel_name"], "nixl_rocm")


if __name__ == "__main__":
    unittest.main()

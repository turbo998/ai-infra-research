from __future__ import annotations

import unittest

from experiments.fa4_fwd_config_sm100_hd256_bwd_verl_thd_logprob_megatron_gdp_qb_mooncake_nixl_dynamo_te_deepep.run import (
    run_simulation as run_fa4_verl_megatron_mooncake,
)
from experiments.flashinfer_cake_fp8_gemm_swap_ab_nvfp4_mla_tcgen05_mxfp8_ep16_push_minimax_h3.run import (
    run_simulation as run_flashinfer_p84,
)
from experiments.sglang_mnnvl_kda_sm107_dcp_qproj_vllm_wk_cublas_nixl_trtllm_gemma4_int8_woq.run import (
    run_simulation as run_sglang_vllm_trtllm_p84,
)


class TestPhase84Experiments(unittest.TestCase):
    def test_flashinfer_cake_fp8_gemm_mla_tcgen05_ep16_minimax_h3(self) -> None:
        res = run_flashinfer_p84()
        self.assertEqual(res["evidence_class"], "own-offline")
        self.assertAlmostEqual(res["grouped_fp8_gemm_swap_ab"]["min_b200_ratio"], 0.581, places=3)
        self.assertEqual(res["nvfp4_sparse_mla_decode_tcgen05"]["total_tcgen05_programs"], 35)
        self.assertAlmostEqual(res["nvfp4_sparse_mla_decode_tcgen05"]["gb300_max_speedup"], 1.375, places=3)
        self.assertEqual(res["mxfp8_megamoe_ep16_push"]["rows_mailbox_bytes_t64"], 6291456)
        self.assertAlmostEqual(
            res["minimax_h3_family"]["sm100_nvfp4_tail_split_p8_m4824"]["gb200_speedup"], 1.049, places=3
        )

    def test_sglang_mnnvl_kda_sm107_dcp_vllm_trtllm(self) -> None:
        res = run_sglang_vllm_trtllm_p84()
        self.assertEqual(res["sglang_kda_sm107"]["cta_multiplier"], 8)
        self.assertAlmostEqual(res["sglang_kda_sm107"]["speedup"], 1.954, places=3)
        self.assertTrue(res["vllm_loader_and_kv_offload"]["fp8_indexer_wk_survives_split_load_weights"])
        self.assertTrue(res["vllm_loader_and_kv_offload"]["nixl_obj_tier_capture_telemetry"])
        tp2_case = [c for c in res["trtllm_gemma4_and_int8_woq"]["int8_woq_tp_cases"] if c["tp_size"] == 4][0]
        self.assertFalse(tp2_case["was_64_aligned"])
        self.assertEqual(tp2_case["padded_per_partition"], 512)

    def test_fa4_verl_megatron_mooncake_nixl_dynamo_te_deepep(self) -> None:
        res = run_fa4_verl_megatron_mooncake()
        self.assertEqual(res["fa4_fwd_and_bwd"]["hd16_max_kv_stages_capped"], 32)
        self.assertTrue(res["verl_thd_and_logprobs"]["raw_mismatch_detected"])
        self.assertEqual(
            res["mooncake_nixl_dynamo_te_deepep"]["tent_rdma_buggy_next_slot_on_255_inflight"], 0
        )
        self.assertEqual(
            res["mooncake_nixl_dynamo_te_deepep"]["tent_rdma_fixed_next_slot_on_255_inflight"], 255
        )
        self.assertAlmostEqual(
            res["mooncake_nixl_dynamo_te_deepep"]["dynamo_atomic_traffic_reduction_x"], 64.0, places=1
        )


if __name__ == "__main__":
    unittest.main()

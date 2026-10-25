"""Unit tests for Phase 86 deterministic Python 3.12 stdlib experiments."""

from __future__ import annotations

import unittest

from experiments.fa4_verl_mcore_nixl_phase86.run import (
    run_all as run_fa4_verl_mcore_nixl,
)
from experiments.flashinfer_softmax_kda_megamoe_phase86.run import (
    run_all as run_flashinfer,
)
from experiments.sglang_vllm_trtllm_phase86.run import (
    run_all as run_sglang_vllm_trtllm,
)


class TestPhase86Experiments(unittest.TestCase):
    def test_flashinfer_softmax_kda_megamoe_phase86(self) -> None:
        res = run_flashinfer()
        self.assertEqual(res["phase"], 86)
        self.assertEqual(res["priority"], 95)
        softmax = res["blackwell_softmax_turnover_ring"]
        self.assertAlmostEqual(softmax["b200_128x128k_speedup"], 1.247, places=3)
        self.assertAlmostEqual(softmax["gb300_128x128k_speedup"], 1.197, places=3)
        kda = res["flashkda_ten_slot_aux_ring"]
        self.assertEqual(kda["five_slot_cross_warp_parity_conflicts"], 15)
        self.assertEqual(kda["ten_slot_cross_warp_parity_conflicts"], 0)
        megamoe = res["sm103a_paced_store_and_megamoe"]
        self.assertEqual(megamoe["megamoe_cutedsl_next_exported_files"], 46)
        self.assertAlmostEqual(megamoe["router_gemm_m16_speedup"], 1.481, places=3)

    def test_sglang_vllm_trtllm_phase86(self) -> None:
        res = run_sglang_vllm_trtllm()
        self.assertEqual(res["phase"], 86)
        sgl = res["sglang_allreduce_and_diffusion"]
        self.assertEqual(sgl["unreg_slices_per_call"], 16)
        self.assertEqual(sgl["leaked_ipc_handles_after_per_step"], 0)
        self.assertAlmostEqual(sgl["diffusion_upload_reduction_ratio"], 50.0, places=1)
        vllm = res["vllm_mrv2_and_rdna3_w4a8"]
        self.assertEqual(vllm["num_write_chunks_dim1"], 3)
        self.assertEqual(vllm["clef_h2d_calls_after"], 1)
        self.assertEqual(vllm["rdna3_v_dot4_instructions_per_group"], 8)
        trt = res["trtllm_stop_words_and_quack_swiglu"]
        self.assertAlmostEqual(trt["stop_buffer_memory_reduction_x"], 3338.8, places=1)
        self.assertEqual(trt["fused_gate_up_intermediate_traffic_mib"], 0)

    def test_fa4_verl_mcore_nixl_phase86(self) -> None:
        res = run_fa4_verl_mcore_nixl()
        self.assertEqual(res["phase"], 86)
        fa4 = res["fa4_1cta_mla_and_te_kv_rope"]
        self.assertEqual(fa4["dq_dqv_spill_bytes_o2"], 0)
        self.assertEqual(fa4["bwd_64h_3stage_tmem_cols"], 448)
        self.assertAlmostEqual(fa4["te_fused_speedup"], 1.363, places=3)
        verl = res["verl_score_centering_and_mcore_moe"]
        self.assertAlmostEqual(verl["renormalized_train_support_sum"], 1.0, places=6)
        self.assertAlmostEqual(verl["residual_sum_over_support"], 0.0, places=7)
        self.assertAlmostEqual(verl["mcore_moe_work_reduction_x"], 12.8, places=2)
        fab = res["mooncake_nixl_dynamo_fabric"]
        self.assertEqual(fab["efa_single_nic_4kb_cap_gib"], 88.0)
        self.assertEqual(fab["nixl_4096_desc_wqes_after"], 2049)
        self.assertAlmostEqual(fab["nixl_throughput_gain_pct"], 22.12, places=2)


if __name__ == "__main__":
    unittest.main()

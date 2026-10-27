"""Unit tests for Phase 101 deterministic Python 3.12 stdlib experiment suites."""

from __future__ import annotations

import unittest

from experiments.dynamo_sidecar_torchtitan_varlen_nixl_phase101.run import (
    run_suite as run_dynamo_suite,
)
from experiments.flashinfer_minimax_h3_tail_kimi_router_phase101.run import (
    run_suite as run_flashinfer_suite,
)
from experiments.vllm_kv_balanced_mla_sglang_aux_phase101.run import (
    run_suite as run_vllm_suite,
)


class TestPhase101Experiments(unittest.TestCase):
    def test_flashinfer_minimax_h3_tail_kimi_router(self) -> None:
        res = run_flashinfer_suite()
        self.assertEqual(res["phase"], 101)
        self.assertEqual(res["priority"], 110)
        hm = res["headline_metrics"]
        self.assertEqual(hm["minimax_bf16_gb300_geomean_speedup"], 1.0343)
        self.assertEqual(hm["minimax_mxfp8_gb300_geomean_speedup"], 1.0325)
        self.assertEqual(hm["minimax_mxfp8_gb300_peak_speedup"], 1.057)
        self.assertEqual(hm["kimi_k3_owner_ctas_required"], 128)
        self.assertEqual(hm["kimi_k3_min_sm_cc100"], 43)
        self.assertEqual(hm["kimi_k3_min_sm_cc103"], 32)
        self.assertEqual(hm["sm120_sage_peak_speedup"], 1.01)

    def test_vllm_kv_balanced_mla_sglang_aux(self) -> None:
        res = run_vllm_suite()
        self.assertEqual(res["phase"], 101)
        self.assertEqual(res["priority"], 110)
        hm = res["headline_metrics"]
        self.assertGreater(hm["mla_planner_speedup_vs_fixed4"], 1.25)
        self.assertEqual(hm["mla_max_planned_requests_h128_q2"], 1024)
        self.assertGreater(hm["sglang_eagle3_vram_saved_mib"], 3900.0)
        self.assertEqual(hm["topk_stale_bin_exposure_with_sync"], 0.0)

    def test_dynamo_sidecar_torchtitan_varlen_nixl(self) -> None:
        res = run_dynamo_suite()
        self.assertEqual(res["phase"], 101)
        self.assertEqual(res["priority"], 110)
        hm = res["headline_metrics"]
        self.assertEqual(hm["dynamo_wire_payload_reduction_pct"], 71.43)
        self.assertGreater(hm["torchtitan_sliding_1024_saving_pct"], 90.0)
        self.assertEqual(hm["nixl_unanchored_ss_false_positives"], 3)
        self.assertEqual(hm["nixl_anchored_ss_false_positives"], 0)
        self.assertEqual(hm["nixl_ephemeral_port_collision_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

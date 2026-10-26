from __future__ import annotations

import unittest

from experiments.flashinfer_cake_fp8_tcgen05_phase100.run import (
    run_experiment as run_flashinfer_phase100,
)
from experiments.sglang_dspark_trtllm_kvcmv2_phase100.run import (
    run_experiment as run_sglang_trtllm_phase100,
)
from experiments.torchtitan_distmuon_nixl_config_phase100.run import (
    run_experiment as run_torchtitan_nixl_phase100,
)


class TestPhase100Experiments(unittest.TestCase):
    def test_flashinfer_cake_fp8_tcgen05_phase100(self) -> None:
        summary = run_flashinfer_phase100()
        self.assertEqual(summary["phase"], 100)
        self.assertEqual(summary["priority"], 109)
        self.assertTrue(all(summary["invariants"].values()))
        att = summary["cake_fp8_sm90_attention"]
        self.assertEqual(att["r1_short_seq_speedup"], 0.86)
        self.assertEqual(att["r2_to_r7_min_speedup"], 1.08)
        self.assertEqual(att["r2_to_r7_max_speedup"], 1.16)
        deslop = summary["tcgen05_and_deslop"]["fp4_allreduce_deslop"]
        self.assertEqual(deslop["cu_files_after"], 49)
        self.assertEqual(deslop["net_line_reduction"], 81881)

    def test_sglang_dspark_trtllm_kvcmv2_phase100(self) -> None:
        summary = run_sglang_trtllm_phase100()
        self.assertEqual(summary["phase"], 100)
        self.assertEqual(summary["priority"], 109)
        self.assertTrue(all(summary["invariants"].values()))
        kvcm = summary["vllm_and_trtllm"]["trtllm_kvcmv2_host_fill"]
        self.assertEqual(kvcm["kInitialFillHelpers"], 7)
        self.assertEqual(kvcm["total_fill_threads"], 8)
        self.assertEqual(kvcm["populate_call_mib"], 4)
        self.assertGreaterEqual(kvcm["fill_speedup"], 6.0)

    def test_torchtitan_distmuon_nixl_config_phase100(self) -> None:
        summary = run_torchtitan_nixl_phase100()
        self.assertEqual(summary["phase"], 100)
        self.assertEqual(summary["priority"], 109)
        self.assertTrue(all(summary["invariants"].values()))
        dm = summary["torchtitan_distmuon"]
        self.assertGreaterEqual(dm["step_speedup"], 14.0)
        self.assertTrue(dm["within_atol_1e_5"])
        nixl = summary["nixl_and_mooncake"]["nixl_backend_config_isolation"]
        self.assertEqual(nixl["fixed_second_backends"], ["UCX"])
        self.assertEqual(nixl["fixed_third_after_clear_backends"], ["UCX"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.mooncake_gid_reprobe_reshard_sim.run import (
    AutoGidCandidateClass,
    AutoGidDataPathAction,
    AutoGidSelection,
    PeerAutoGidStateTableSim,
    simulate_bounded_scatter_reshard,
    simulate_gid_reprobe_trace,
)
from experiments.te_grouped_mxfp8_swizzle_sim.run import (
    find_tensor,
    grouped_mxfp8_scale_bytes_upper_bound,
    simulate_grouped_prefix_sum_and_tiles,
    simulate_scenarios,
    table_smem_bytes,
)
from experiments.verl_fp4_refit_tail_kl_sim.run import (
    compute_tail_aware_logit_gradient_1d,
    compute_tail_bucket_kl_scalar,
    simulate_fp4_moe_refit_staging,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase24Experiments(unittest.TestCase):
    def test_te_grouped_mxfp8_upper_bound_and_smem_optin(self) -> None:
        summary = simulate_scenarios()
        scenarios = {s["scenario"]: s for s in summary["scenarios"]}
        for s in summary["scenarios"]:
            self.assertGreaterEqual(s["host_upper_bound_bytes"], s["exact_mxfp8_scale_bytes"])
            self.assertEqual(s["exact_mxfp8_scale_bytes"], s["total_128x128_tiles"] * 512)

        self.assertEqual(scenarios["ds_v3_256e_1d_fprop"]["table_smem_bytes"], 6152)
        self.assertFalse(scenarios["ds_v3_256e_1d_fprop"]["requires_optin_smem"])
        self.assertEqual(scenarios["ds_v3_256e_2d_fprop"]["scale_bytes_vs_input_ratio"], 128.0)
        self.assertEqual(scenarios["ultra_moe_2048e_optin_smem"]["table_smem_bytes"], 49160)
        self.assertTrue(scenarios["ultra_moe_2048e_optin_smem"]["requires_optin_smem"])

    def test_te_find_tensor_binary_search_with_zero_row_experts(self) -> None:
        rows = [0, 0, 128, 0, 256, 0, 64]
        cols = [256] * len(rows)
        sim = simulate_grouped_prefix_sum_and_tiles(rows, cols, is_2d=False)
        # Tiles per tensor: [0, 0, 2, 0, 4, 0, 2] -> total = 8
        self.assertEqual(sim["total_tiles"], 8)
        tile_start = [0, 0, 0, 2, 2, 6, 6, 8]
        self.assertEqual(find_tensor(tile_start, len(rows), 0), 2)
        self.assertEqual(find_tensor(tile_start, len(rows), 1), 2)
        self.assertEqual(find_tensor(tile_start, len(rows), 2), 4)
        self.assertEqual(find_tensor(tile_start, len(rows), 5), 4)
        self.assertEqual(find_tensor(tile_start, len(rows), 6), 6)

    def test_te_grouped_mxfp8_cli_check(self) -> None:
        script = REPO_ROOT / "experiments" / "te_grouped_mxfp8_swizzle_sim" / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CHECK PASSED", proc.stdout)
        summary_file = script.parent / "results" / "summary.json"
        self.assertTrue(summary_file.exists())

    def test_mooncake_gid_reprobe_two_endpoint_guard_and_advisory(self) -> None:
        trace = simulate_gid_reprobe_trace()
        self.assertEqual(trace["first_failure_action"], "kRetrySameGid")
        self.assertEqual(trace["duplicate_endpoint_action"], "kIgnoreDuplicate")
        self.assertEqual(trace["second_endpoint_failure_action"], "kAdvanceGid")
        self.assertEqual(trace["advanced_gid_index"], 2)
        self.assertEqual(trace["advanced_candidate_class"], "kNetworkPrivateV4")
        self.assertEqual(trace["stale_epoch_action"], "kIgnoreDuplicate")
        self.assertEqual(trace["post_recovery_remaining_advances"], 2)
        self.assertEqual(trace["advisory_override_when_exhausted_gid_index"], 2)

    def test_mooncake_bounded_scatter_reshard_window(self) -> None:
        rows = simulate_bounded_scatter_reshard()
        by_w = {r["max_inflight_batches"]: r for r in rows}
        self.assertEqual(by_w[1]["peak_inflight_observed"], 1)
        self.assertEqual(by_w[4]["peak_inflight_observed"], 4)
        self.assertGreater(by_w[4]["effective_goodput_gbps"], by_w[1]["effective_goodput_gbps"])
        self.assertEqual(by_w[4]["pinned_window_footprint_mib"], 1024)
        self.assertEqual(by_w[4]["submission_state_on_unknown_cqe"], "POISONED")

    def test_mooncake_gid_reprobe_cli_check(self) -> None:
        script = REPO_ROOT / "experiments" / "mooncake_gid_reprobe_reshard_sim" / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CHECK PASSED", proc.stdout)

    def test_verl_tail_kl_non_negativity_and_analytic_gradient(self) -> None:
        res = compute_tail_aware_logit_gradient_1d(
            student_probs=[0.52, 0.40, 0.02, 0.015, 0.015, 0.01, 0.01, 0.01],
            teacher_topk_probs=[0.40, 0.30],
            teacher_topk_indices=[0, 1],
            tail_mass_eps=1e-6,
        )
        self.assertLess(res["raw_topk_kl"], 0.0)
        self.assertEqual(res["clamped_topk_kl"], 0.0)
        self.assertGreater(res["coarse_grained_total_kl"], 0.20)
        self.assertLess(res["max_finite_diff_error"], 1e-6)
        self.assertLess(res["grad_sum_zero_invariant"], 1e-9)
        # Zero teacher tail contributes zero by continuity
        self.assertAlmostEqual(compute_tail_bucket_kl_scalar(0.85, 1.0, 1e-6), 0.0, places=9)

    def test_verl_fp4_zero_copy_refit_staging_ratios(self) -> None:
        rows = {r["refit_scheme"]: r for r in simulate_fp4_moe_refit_staging()}
        self.assertEqual(rows["verl_MXFP4xMXFP8_zero_copy_alias"]["staging_to_weight_ratio"], 0.0625)
        self.assertEqual(rows["verl_NVFP4_W4A4_zero_copy_alias"]["staging_to_weight_ratio"], 0.125)
        self.assertTrue(rows["verl_MXFP4xMXFP8_zero_copy_alias"]["cuda_graph_addresses_preserved"])
        self.assertFalse(rows["FP4_naive_duplicate_staging"]["cuda_graph_addresses_preserved"])

    def test_verl_fp4_refit_tail_kl_cli_check(self) -> None:
        script = REPO_ROOT / "experiments" / "verl_fp4_refit_tail_kl_sim" / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CHECK PASSED", proc.stdout)


if __name__ == "__main__":
    unittest.main()

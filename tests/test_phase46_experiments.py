from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase46Experiments(unittest.TestCase):
    def _run_and_load(self, exp_name: str) -> dict:
        script = REPO_ROOT / "experiments" / exp_name / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("check=True", res.stdout)
        summary_path = script.with_name("summary.json")
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_cake_allgather_hopper_msa_vsa_tmem_race(self) -> None:
        data = self._run_and_load("flashinfer_cake_allgather_hopper_msa_vsa_tmem_race")
        self.assertEqual(data["phase"], "Phase 46")
        self.assertEqual(data["priority"], 55)
        self.assertEqual(data["headline_metrics"]["cake_msa_sm90_total_programs"], 89)
        self.assertEqual(data["headline_metrics"]["cake_vsa_sm90_total_programs"], 17)
        self.assertEqual(data["headline_metrics"]["dsa_train_fixed_tmem_lanes_covered"], 128)
        self.assertEqual(data["headline_metrics"]["cta_pair_drained_kernels_total"], 50)

    def test_trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu(self) -> None:
        data = self._run_and_load("trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu")
        self.assertEqual(data["phase"], "Phase 46")
        self.assertEqual(data["priority"], 55)
        self.assertEqual(data["headline_metrics"]["swaps_vs_legacy_mask_capacity_ratio"], 4.0)
        self.assertEqual(data["headline_metrics"]["kvcm2_64gib_startup_commit_reduction_x"], 2048.0)
        self.assertEqual(data["headline_metrics"]["hy_v4_ihc_hbm_traffic_reduction_ratio"], 1.6)
        self.assertEqual(data["headline_metrics"]["sglang_prefill_tune_max_num_tokens_cap"], 8192)

    def test_nixl_gpunetio_read_ucx_proxy_mooncake_tent_dynamo_planner_torchtitan_te(self) -> None:
        data = self._run_and_load(
            "nixl_gpunetio_read_ucx_proxy_mooncake_tent_dynamo_planner_torchtitan_te"
        )
        self.assertEqual(data["phase"], "Phase 46")
        self.assertEqual(data["priority"], 55)
        self.assertEqual(data["headline_metrics"]["gpunetio_wqe_reduction_x"], 2048)
        self.assertEqual(data["headline_metrics"]["gpunetio_ring_entry_reduction_x"], 4)
        self.assertAlmostEqual(data["headline_metrics"]["gpunetio_h20_throughput_gain_pct"], 109.92)
        self.assertEqual(data["headline_metrics"]["mooncake_tent_request_fixed_sizeof"], 48)


if __name__ == "__main__":
    unittest.main()

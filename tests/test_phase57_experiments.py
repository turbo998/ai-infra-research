"""Unit tests for Phase 57 (priority: 66) deterministic CPU/offline experiments."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase57Experiments(unittest.TestCase):
    def _run_experiment(self, slug: str) -> dict[str, object]:
        exp_dir = REPO_ROOT / "experiments" / slug
        script = exp_dir / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS]", proc.stdout)
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.is_file())
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(data["all_invariants_passed"])
        return data

    def test_track_a_flashinfer_phase57(self) -> None:
        data = self._run_experiment(
            "flashinfer-megamoe-topk-sm103-ag-svdquant-kda-weakref-mxfp4-k128"
        )
        self.assertEqual(
            data["kda_weakref_checkpoint_and_unzeroed_arena"]["h16_retained_after_mib"],
            0.0,
        )
        self.assertEqual(
            data["hopper_mxfp4_fp8_k128_candidates_and_autotuner"]["eligible_i640_after"],
            8,
        )

    def test_track_b_sglang_trtllm_vllm_phase57(self) -> None:
        data = self._run_experiment(
            "sglang-rust-abort-mm-finalize-trtllm-swiglu-nvfp4-vllm-topk-gdn"
        )
        self.assertEqual(
            data["trtllm_sm100_swiglu_and_nvfp4_sync_prefill"]["async_input_quant_max_tokens"],
            1024,
        )
        self.assertEqual(
            data["vllm_cooperative_topk_and_cpu_gdn"]["new_path"],
            "single_pass_staging",
        )

    def test_track_c_megatron_mooncake_nixl_phase57(self) -> None:
        data = self._run_experiment(
            "megatron-nvfp4-moe-mfsdp-planes-pad-zero-mooncake-xpu-nof-nixl"
        )
        self.assertEqual(
            data["megatron_nvfp4_mfsdp_and_pad_zeroing"]["mfsdp_v2_mxfp8_fwd_allgather_reduction_ratio"],
            0.5,
        )
        self.assertEqual(
            data["nixl_v6_ring_buffer_and_dynamo_relay"]["header_total_bytes"],
            512,
        )


if __name__ == "__main__":
    unittest.main()

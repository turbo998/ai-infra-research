"""Unit tests for Phase 52 (priority: 61) deterministic experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPhase52Experiments(unittest.TestCase):
    def _run_experiment(self, slug: str) -> dict:
        script = ROOT / "experiments" / slug / "run.py"
        summary_path = ROOT / "experiments" / slug / "summary.json"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS]", res.stdout)
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_cake_vsa_sm90_powlu_moe_qk_rope_append_dcp_one_wave_megamoe(self) -> None:
        data = self._run_experiment(
            "flashinfer-cake-vsa-sm90-powlu-moe-qk-rope-append-dcp-one-wave-megamoe"
        )
        self.assertEqual(data["phase"], 52)
        self.assertEqual(data["priority"], 61)
        self.assertEqual(data["cake_vsa_sm90"]["total_programs"], 17)
        self.assertEqual(data["cake_vsa_sm90"]["faster_rows"], 119)
        self.assertEqual(data["powlu_moe"]["default_powlu_m"], 2.5)
        self.assertEqual(data["qk_rope_append_and_dcp"]["dcp_fp8_d256_gb300_speedup"], 1.24)
        self.assertEqual(data["qk_rope_append_and_dcp"]["sm120_manifest_seals_verified"], 18)

    def test_sglang_mi35x_ibv_tool_coercion_trtllm_fp8_custom_mask_vllm_rust_cli(self) -> None:
        data = self._run_experiment(
            "sglang-mi35x-ibv-tool-coercion-trtllm-fp8-custom-mask-vllm-rust-cli"
        )
        self.assertEqual(data["phase"], 52)
        self.assertEqual(data["sglang"]["kv_registration_path_gfx950"], "ibv_reg_mr")
        self.assertEqual(data["sglang"]["coercion_cases"]["whole_float_str"], 3)
        self.assertEqual(data["sglang"]["coercion_cases"]["fractional_float_str"], "3.5")
        self.assertEqual(data["trtllm"]["fp8_clamped_out"], 448.0)
        self.assertEqual(data["vllm_rust"]["zero_interval_clamped_ns"], 1)

    def test_te_zero_wgrad_argmax_torchtitan_clip_hf_cursor_lmcache_nixl_mooncake(self) -> None:
        data = self._run_experiment(
            "te-zero-wgrad-argmax-torchtitan-clip-hf-cursor-lmcache-nixl-mooncake"
        )
        self.assertEqual(data["phase"], 52)
        self.assertEqual(data["transformer_engine"]["efficient_zero_storage_bytes"], 0)
        self.assertEqual(data["transformer_engine"]["nan_after_max_global_argmax"], 5)
        self.assertTrue(data["torchtitan"]["after_fix_state_preserved"])
        self.assertEqual(data["torchtitan"]["after_fix_resaved_resume_row"], 3)
        self.assertEqual(data["kv_and_fabric"]["nixl_ucx_proxy_workers"], 128)


if __name__ == "__main__":
    unittest.main()

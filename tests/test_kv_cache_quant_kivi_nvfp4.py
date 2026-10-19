"""Unit tests for Phase 17 experiment: `kv_cache_quant_kivi_nvfp4`."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest

from experiments.kv_cache_quant_kivi_nvfp4.run import (
    deterministic_rht_signs,
    pack_scale_with_zp_steganography,
    run_simulation,
    single_rht,
    validate_summary,
)


class TestKvCacheQuantKiviNvfp4(unittest.TestCase):
    def test_rht_invertibility(self) -> None:
        d = 128
        signs = deterministic_rht_signs(d)
        vec = [float((i % 11) - 5) * 0.37 for i in range(d)]
        rot = single_rht(vec, signs, inverse=False)
        rec = single_rht(rot, signs, inverse=True)
        for a, b in zip(vec, rec):
            self.assertAlmostEqual(a, b, places=6)

    def test_zero_point_steganography_exact_recovery(self) -> None:
        for zp in range(16):
            _, unpacked_s, unpacked_zp = pack_scale_with_zp_steganography(0.09375, zp)
            self.assertEqual(unpacked_zp, zp)
            self.assertAlmostEqual(unpacked_s, 0.09375, places=5)

    def test_quantization_hierarchy_and_cli_check(self) -> None:
        summary = run_simulation()
        validate_summary(summary)
        q = summary["quant_mode_comparison"]
        self.assertLess(
            q["int4_per_token_head_rht_stego"]["attention_score_kl_divergence"],
            q["int4_per_token_head_no_rht"]["attention_score_kl_divergence"] * 0.1,
        )

        script = (
            pathlib.Path(__file__).resolve().parents[1]
            / "experiments"
            / "kv_cache_quant_kivi_nvfp4"
            / "run.py"
        )
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] kv_cache_quant_kivi_nvfp4", proc.stdout)
        summary_file = script.parent / "results" / "summary.json"
        loaded = json.loads(summary_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["Check"], "PASS")


if __name__ == "__main__":
    unittest.main()

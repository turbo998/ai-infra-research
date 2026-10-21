import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase49Experiments(unittest.TestCase):
    def _run_experiment(self, slug: str) -> dict:
        script = ROOT / "experiments" / slug / "run.py"
        result = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.returncode, 0)
        summary_path = ROOT / "experiments" / slug / "summary.json"
        self.assertTrue(summary_path.is_file())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_topic1_flashinfer_frost_cake_rmsnorm_sage_vsa(self) -> None:
        summary = self._run_experiment(
            "flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa"
        )
        self.assertEqual(summary["phase"], 49)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertAlmostEqual(
            summary["frost_low_latency_moe"]["peak_bf16_speedup"], 1.794, places=3
        )
        self.assertAlmostEqual(
            summary["frost_low_latency_moe"]["peak_mx_speedup"], 1.608, places=3
        )
        self.assertEqual(
            summary["cake_sage_ssd_vsa"]["sage_smem_delta_bytes"], 1280
        )
        self.assertEqual(
            summary["cake_sage_ssd_vsa"]["total_vsa_sm90_programs"], 18
        )

    def test_topic2_vllm_fused_mm_norm_rocm_hyv4_scaleout_derender(self) -> None:
        summary = self._run_experiment(
            "vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang"
        )
        self.assertEqual(summary["phase"], 49)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertFalse(summary["fused_mm_input_norm"]["persistent_buffer"])
        self.assertEqual(
            summary["rust_frontend_trtllm_sglang"]["num_bucket_families"], 8
        )
        self.assertEqual(
            summary["rust_frontend_trtllm_sglang"]["trtllm_max_video_bytes"],
            200 * 1024 * 1024,
        )

    def test_topic3_lmcache_dynamo_nixl_te_jax_torchtitan(self) -> None:
        summary = self._run_experiment(
            "lmcache-mp-module-dynamo-nixl-te-jax-moe-torchtitan-rl-pp"
        )
        self.assertEqual(summary["phase"], 49)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertAlmostEqual(
            summary["megatron_te_jax_torchtitan"]["te_jax_expected_swiglu"],
            2.193176,
            places=6,
        )
        self.assertEqual(
            summary["megatron_te_jax_torchtitan"]["torchtitan_dp0_slice"],
            [0, 1, 4, 5],
        )


if __name__ == "__main__":
    unittest.main()

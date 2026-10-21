import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPhase55Experiments(unittest.TestCase):
    def _run_and_load(self, slug: str) -> dict:
        exp_dir = ROOT / "experiments" / slug
        script = exp_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        return json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))

    def test_flashinfer_cake_kda_tf32_fp8_gemm_stepfun_moe_minimax_h3_cudnn_hnd(self) -> None:
        data = self._run_and_load(
            "flashinfer-cake-kda-tf32-fp8-gemm-stepfun-moe-minimax-h3-cudnn-hnd"
        )
        self.assertTrue(data["all_invariants_passed"])
        self.assertEqual(data["cake_kda_tf32"]["lines_before"], 97009)
        self.assertEqual(data["cake_kda_tf32"]["lines_after"], 41231)
        self.assertEqual(data["cake_kda_tf32"]["canonical_bodies_after"], 9)
        self.assertEqual(data["cake_fp8_and_stepfun_moe"]["fp8_total_programs"], 60)
        self.assertEqual(data["cake_fp8_and_stepfun_moe"]["stepfun_manifest_entries"], 222)
        self.assertFalse(data["minimax_h3_and_cudnn_hnd"]["buggy_nhd_gqa_valid"])
        self.assertTrue(data["minimax_h3_and_cudnn_hnd"]["fixed_hnd_gqa_valid"])

    def test_sglang_extend_host_prompt_reuseport_rust_probe_trtllm_v2_vllm_stream(self) -> None:
        data = self._run_and_load(
            "sglang-extend-host-prompt-reuseport-rust-probe-trtllm-v2-vllm-stream"
        )
        self.assertTrue(data["all_invariants_passed"])
        self.assertFalse(data["sglang_extend_and_mamba"]["chunk_only_completes_prompt"])
        self.assertTrue(data["sglang_extend_and_mamba"]["completing_extend_completes_prompt"])
        self.assertEqual(
            data["sglang_extend_and_mamba"]["mamba_gathered_track_indices"], [41, 20, 11]
        )
        self.assertEqual(
            data["sglang_reuseport_and_rust_probe"]["probes_answered_by_single_ack"], 4
        )
        self.assertEqual(
            data["trtllm_and_vllm_frontend"]["suppressed_when_parallel_false"], 2
        )

    def test_te_fusible_deferred_init_dsink_titan_ulysses_lmcache_fs_mooncake_nixl(self) -> None:
        data = self._run_and_load(
            "te-fusible-deferred-init-dsink-titan-ulysses-lmcache-fs-mooncake-nixl"
        )
        self.assertTrue(data["all_invariants_passed"])
        self.assertEqual(
            data["te_and_torchtitan"]["te_thd_dsink_allowed_dims_after"], [64, 128, 256]
        )
        self.assertEqual(
            data["te_and_torchtitan"]["new_fixed_decoded"], "<think>reason</think>answer"
        )
        self.assertEqual(
            data["lmcache_mooncake_nixl"]["lmcache_fs_default_read_budget_bytes"], 33554432
        )
        self.assertEqual(
            data["lmcache_mooncake_nixl"]["mooncake_efa_bilateral_deadlocks_after"], 0
        )
        self.assertEqual(
            data["lmcache_mooncake_nixl"]["nixl_gpunetio_new_bounded_len"], 16
        )


if __name__ == "__main__":
    unittest.main()

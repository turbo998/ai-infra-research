"""Unit tests for Phase 44 (priority: 53) deterministic experiments."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase44Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        script = ROOT / rel_dir / "run.py"
        summary_path = ROOT / rel_dir / "summary.json"
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=ROOT)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_masked_mxfp8_sm90_megamoe_bmm_fp8(self) -> None:
        data = self._run_and_load("experiments/flashinfer_masked_mxfp8_sm90_megamoe_bmm_fp8")
        self.assertTrue(data["all_invariants_passed"])
        p1 = data["masked_mxfp8_experts_quantize"]
        self.assertEqual(p1["total_valid_rows"], 1856)
        self.assertEqual(p1["b200_benchmarks"]["gemm1_k7168_valid16"]["speedup_vs_cutile"], 10.03)
        self.assertEqual(p1["b200_benchmarks"]["silu_mul_2x2048_valid16"]["speedup_vs_cutile"], 18.69)
        p2 = data["sm90_megamoe_fixes_and_push_policy"]
        self.assertEqual(p2["dirty_blocks_unfixed"], 112)
        self.assertEqual(p2["dirty_blocks_fixed"], 0)
        self.assertEqual(p2["global_scale_pow2"], 8.0)
        p3 = data["blackwell_bmm_fp8_and_mxfp8_one_wave"]
        self.assertEqual(p3["max_one_wave_n"], 2368)
        self.assertEqual(p3["pdl_benchmark_summary"]["B300_150_cells"]["regressions_after"], 0)

    def test_sglang_mooncake_probe_glm52_trtllm_glm53_sparse(self) -> None:
        data = self._run_and_load("experiments/sglang_mooncake_probe_glm52_trtllm_glm53_sparse")
        self.assertTrue(data["all_invariants_passed"])
        p1 = data["sglang_mooncake_probe_and_nixl_ucx"]
        self.assertEqual(p1["probes_before_idle_prune"], 4)
        self.assertEqual(p1["pruned_at_s"], 750.0)
        self.assertEqual(p1["valid_device_list"], "mlx5_0, mlx5_1")
        p3 = data["trtllm_glm53_kpool_and_sm107_mxfp8"]
        self.assertEqual(p3["executed_pool_updates"], 3)
        self.assertTrue(p3["fp8_tp1_64heads_native_new"])
        self.assertIn(107, p3["mxfp8_sm_constraint_after"])

    def test_megatron_mfsdp_v2_exec_order_grad_reuse_mooncake_cursor(self) -> None:
        data = self._run_and_load("experiments/megatron_mfsdp_v2_exec_order_grad_reuse_mooncake_cursor")
        self.assertTrue(data["all_invariants_passed"])
        p1 = data["megatron_mfsdp_v2_and_grad_copy"]
        self.assertTrue(p1["divergence_caught"])
        self.assertEqual(p1["saved_vram_gb"], 4.0)
        self.assertEqual(p1["te_linear_attn_dynamo_graph_breaks_after"], 0)
        p2 = data["mooncake_shard_cursor_and_cuda_ctx"]
        self.assertTrue(p2["clear_flag_preserved"])
        self.assertIsNone(p2["bare_thread_case"]["new_restored_ctx"])
        self.assertEqual(p2["custom_ctx_case"]["new_restored_ctx"], "custom_ctx_gpu0")


if __name__ == "__main__":
    unittest.main()

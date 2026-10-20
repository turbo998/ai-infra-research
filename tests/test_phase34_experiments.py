import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase34Experiments(unittest.TestCase):
    def _run_experiment(self, rel_dir: str) -> dict:
        script = REPO_ROOT / rel_dir / "run.py"
        summary_path = REPO_ROOT / rel_dir / "results" / "summary.json"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"status": "ok"', res.stdout)
        import json

        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_vllm_trtllm_dcp_symm_scratch_megatron_scale(self) -> None:
        data = self._run_experiment("experiments/vllm_trtllm_dcp_symm_scratch_megatron_scale")
        dcp = data["trtllm_dcp_prefill"]
        self.assertTrue(dcp["workspace_checks"][0]["fits_trtllm_workspace"])
        self.assertFalse(dcp["workspace_checks"][2]["fits_trtllm_workspace"])
        self.assertTrue(dcp["empty_shard_has_nan_without_zero_and_mask"])
        self.assertEqual(dcp["empty_shard_abs_err"], 0.0)
        symm = data["symm_mem_scratch"]
        self.assertEqual(symm["buggy_overwrite_l1_error"], 30.75)
        self.assertEqual(symm["fixed_overwrite_l1_error"], 0.0)
        sc = data["megatron_softmax_scale"]
        self.assertEqual(sc["std_attention_fixed_scale"], 0.095)
        self.assertGreater(sc["fixed_vs_legacy_scale_ratio"], 1.31)

    def test_torchtitan_compiled_rope_swiglu_fp32_rmsnorm(self) -> None:
        data = self._run_experiment("experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm")
        rope = data["compiled_complex_rope"]
        self.assertEqual(rope["int32_pair_max_ulp_diff_vs_complex"], 0)
        self.assertTrue(rope["inverse_rotation_consistent"])
        self.assertEqual(rope["autotune_on_distinct_losses_in_13_runs"], 9)
        self.assertEqual(rope["autotune_off_distinct_losses_in_12_runs"], 1)
        sw = data["swiglu_unmasked_cat"]
        self.assertEqual(sw["gb300_swiglu_table"][2]["delta_pct"], -27.24)
        rn = data["muse_spark_fp32_rmsnorm"]
        self.assertEqual(rn["max_abs_diff_fp32_cast_vs_fallback"], 0.0)
        self.assertEqual(rn["decomposed_fallback_kernel_count_fwd_bwd"], 20)
        self.assertEqual(rn["fused_fwd_kernel_count"], 1)

    def test_trtllm_kvcm_v2_retention_lmcache_l1_tent_notify(self) -> None:
        data = self._run_experiment("experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify")
        kvcm = data["trtllm_kvcm_v2_retention"]
        self.assertEqual(kvcm["cold_tier_write_reduction_pct"], 50.0)
        self.assertEqual(kvcm["helix_cp_scaled_global_range"], [0, 1024, 85])
        lmc = data["lmcache_l1_orchestrator"]
        self.assertEqual(lmc["retired_writes_on_inc2_register"], 2)
        self.assertFalse(lmc["oversized_batch_allocated_any"])
        self.assertEqual(lmc["idempotent_identical_retry"], "REPLAYED_OK")
        self.assertEqual(lmc["idempotent_mutated_retry"], "ALREADY_EXISTS")
        tent = data["mooncake_tent_notify"]
        self.assertEqual(tent["silent_false_acks_legacy"], 3)
        self.assertEqual(tent["silent_false_acks_fixed"], 0)


if __name__ == "__main__":
    unittest.main()

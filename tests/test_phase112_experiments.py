import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPhase112Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> dict:
        script = ROOT / rel_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("passed", res.stdout)
        summary_path = ROOT / rel_dir / "summary.json"
        self.assertTrue(summary_path.is_file())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_rubin_megamoe_sm120_l2_cutlass_tilelang(self) -> None:
        data = self._run_suite(
            "experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112"
        )
        self.assertEqual(data["phase"], 112)
        self.assertEqual(data["track"], "A")
        self.assertTrue(data["rubin_megamoe_route_staging"]["fixed_check_passed"])
        self.assertTrue(data["rubin_megamoe_route_staging"]["masked_routes_zeroed"])
        self.assertGreater(data["sm120_mxfp8_mxfp4_tma_l2_promotion"]["AuthorMeasured_speedup"], 1.35)
        self.assertEqual(data["cake_minimax_h3_and_qsa_sm107"]["fc1_w448_n_tiles"], 64)
        self.assertEqual(data["strided_plan_cute_ir_and_tilelang"]["reconciled_sibling_buffer_ndim"], 3)

    def test_track_b_sglang_paged_experts_vllm_ds41_fp8_trtllm_sm107(self) -> None:
        data = self._run_suite(
            "experiments/sglang_paged_experts_vllm_ds41_fp8_indexer_trtllm_sm107_phase112"
        )
        self.assertEqual(data["phase"], 112)
        self.assertEqual(data["track"], "B")
        self.assertEqual(
            data["sglang_paged_experts_and_diffusion"]["d2h_sync_reduction_ratio"], 4.0
        )
        self.assertEqual(
            data["vllm_prefix_ds41_mooncake_checkpoint"]["prefix_hit_start5_local"], 4
        )
        self.assertEqual(
            data["vllm_prefix_ds41_mooncake_checkpoint"]["ds41_64k_logit_work_reduction_x"],
            16.0,
        )
        self.assertTrue(data["trtllm_sm107_and_low_m_gemm"]["fixed_captured_logits_distinct"])

    def test_track_c_te_compile_megatron_mfsdp_torchtitan_mooncake_nixl(self) -> None:
        data = self._run_suite(
            "experiments/te_compile_fused_attn_megatron_mfsdp_mooncake_tent_nixl_phase112"
        )
        self.assertEqual(data["phase"], 112)
        self.assertEqual(data["track"], "C")
        self.assertTrue(data["te_compile_and_dglu"]["dglu_deterministic_dprob_supported"])
        self.assertEqual(
            data["megatron_mfsdp_and_torchtitan_ema"]["mfsdp_mxfp8_events_per_group"], 2
        )
        self.assertTrue(
            data["megatron_mfsdp_and_torchtitan_ema"]["fixed_base_ckpt_ema_matches_model"]
        )
        self.assertTrue(
            data["mooncake_lmcache_dynamo_nixl"]["nixl_posix_fixed_completes_rollback"]
        )


if __name__ == "__main__":
    unittest.main()

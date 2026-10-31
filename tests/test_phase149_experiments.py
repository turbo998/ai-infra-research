import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase149Experiments(unittest.TestCase):
    def test_exp1_flashinfer_triton_tilelang(self):
        exp = ROOT / "experiments" / "flashinfer_mxfp8_band_mok_kg1_minimax_phase149" / "run.py"
        subprocess.run([sys.executable, str(exp), "--check"], check=True)
        res = json.loads(exp.with_name("results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["phase"], 149)
        self.assertGreater(res["mxfp8_band_backward"]["band_speedup_ratio"], 1.2)
        self.assertLess(res["mok_rubin_ep8"]["mok_step_us_tuned"], res["mok_rubin_ep8"]["mok_step_us_legacy"])
        self.assertGreater(res["grouped_fp8_gemm_kg1"]["scale_traffic_reduction_pct"], 80.0)
        self.assertEqual(res["tilelang_nested_ldg_guard"]["predicated_oob_lanes"], 0)

    def test_exp2_sglang_vllm_trtllm(self):
        exp = ROOT / "experiments" / "sglang_aiter_xpu_vllm_rust_trtllm_tpot_phase149" / "run.py"
        subprocess.run([sys.executable, str(exp), "--check"], check=True)
        res = json.loads(exp.with_name("results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["phase"], 149)
        self.assertGreater(res["sglang_amd_aiter_fav3_gfx942"]["gfx942_fav3_speedup"], 1.5)
        self.assertGreater(res["sglang_intel_xpu_page_size_1"]["kv_waste_reduction_pct"], 3.0)
        self.assertFalse(res["trtllm_tpot_and_lifecycle"]["buggy_flags_slow_regression"])
        self.assertTrue(res["trtllm_tpot_and_lifecycle"]["fixed_flags_slow_regression"])

    def test_exp3_megatron_mooncake_dynamo_nixl(self):
        exp = ROOT / "experiments" / "megatron_wandb_mooncake_dynamo_nixl_phase149" / "run.py"
        subprocess.run([sys.executable, str(exp), "--check"], check=True)
        res = json.loads(exp.with_name("results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["phase"], 149)
        self.assertEqual(res["megatron_wandb_hygiene"]["daily_wandb_upload_gb_fixed"], 0.0)
        self.assertEqual(res["mooncake_quota_and_tent_failover"]["fixed_max_admitted_inflight"], 256)
        self.assertEqual(res["mooncake_quota_and_tent_failover"]["transfer_success_rate_fixed"], 1.0)
        self.assertTrue(res["dynamo_bounded_pinned_scoring"]["pinned_selected"])


if __name__ == "__main__":
    unittest.main()

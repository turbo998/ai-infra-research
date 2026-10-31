import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase151Experiments(unittest.TestCase):
    def test_track1_flashinfer_triton_tilelang_phase151(self):
        exp_dir = ROOT / "experiments" / "flashinfer_lmhead_minimax_kda_mxfp8_fmha_phase151"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 151)
        self.assertEqual(data["cake_lm_head_budget_and_valid_rows"]["lm_head_flops_saving_pct"], 37.5)
        self.assertGreater(data["sm120_varlen_and_kda_n32"]["sm120_varlen_speedup"], 1.25)
        self.assertEqual(data["triton_gsan_and_tilelang_codegen"]["triton_class_constexpr_preload_failures_fixed"], 0)

    def test_track2_sglang_vllm_trtllm_phase151(self):
        exp_dir = ROOT / "experiments" / "sglang_paged_experts_mhc_vllm_w4a16_trtllm_b12x_phase151"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 151)
        self.assertTrue(data["vllm_rocm_w4a16_int64_offsets"]["int32_overflow_triggered"])
        self.assertTrue(data["vllm_rocm_w4a16_int64_offsets"]["int64_offset_safe"])
        self.assertGreater(data["vllm_muse_spark_and_epd_metadata"]["muse_spark_routing_speedup"], 3.5)

    def test_track3_megatron_mooncake_dynamo_nixl_phase151(self):
        exp_dir = ROOT / "experiments" / "megatron_gdn_fp64_mooncake_uring_dynamo_nixl_phase151"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 151)
        self.assertEqual(data["megatron_gdn_and_batch_invariant_fp64"]["moe_combine_fp64_drift_vs_train"], 0.0)
        self.assertEqual(data["dynamo_router_and_nixl_libfabric_lock"]["nixl_libfabric_ep_mutex_races_fixed"], 0)
        self.assertGreater(data["mooncake_tent_and_lmcache_pd"]["tent_lookup_speedup"], 7.0)


if __name__ == "__main__":
    unittest.main()

import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase150Experiments(unittest.TestCase):
    def test_exp1_flashinfer_triton_tilelang(self):
        exp = ROOT / "experiments" / "flashinfer_ssd_warpdecode_nvfp4_mla_sm120_phase150" / "run.py"
        subprocess.run([sys.executable, str(exp), "--check"], check=True)
        res = json.loads(exp.with_name("results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["phase"], 150)
        self.assertGreater(res["sm107a_ssd_combined"]["ssd_speedup_ratio"], 1.25)
        self.assertGreater(res["sm100_warp_decode_fc1_fc2"]["warp_decode_speedup"], 1.30)
        self.assertFalse(res["compiler_safety_and_jit"]["fixed_matches_neg_zero_as_pos_zero"])

    def test_exp2_sglang_vllm_trtllm(self):
        exp = ROOT / "experiments" / "sglang_dspark_mla_vllm_umbp_trtllm_openengine_phase150" / "run.py"
        subprocess.run([sys.executable, str(exp), "--check"], check=True)
        res = json.loads(exp.with_name("results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["phase"], 150)
        self.assertGreater(res["sglang_mla_kscale_skip"]["mla_kscale_skip_speedup"], 1.20)
        self.assertEqual(res["sglang_npu_dspark_verify_kv"]["eliminated_phantom_kv_slots"], 60)
        self.assertEqual(res["vllm_preload_and_http_resilience"]["orphan_nccl_groups_fixed"], 0)

    def test_exp3_megatron_mooncake_dynamo_nixl(self):
        exp = ROOT / "experiments" / "megatron_ckpt_shortcut_mooncake_ha_dynamo_nixl_phase150" / "run.py"
        subprocess.run([sys.executable, str(exp), "--check"], check=True)
        res = json.loads(exp.with_name("results.json").read_text(encoding="utf-8"))
        self.assertEqual(res["phase"], 150)
        self.assertEqual(res["megatron_ckpt_and_shortcut_moe"]["newer_ckpt_deleted_fixed"], 0)
        self.assertEqual(res["mooncake_ha_and_replica_probe"]["ha_false_stepdowns_fixed"], 0)
        self.assertGreater(res["dynamo_admission_and_cancellation"]["gpu_waste_reduction_pct"], 95.0)
        self.assertEqual(res["nixl_obj_duplicate_devid_guard"]["duplicate_rejected_count"], 1)


if __name__ == "__main__":
    unittest.main()

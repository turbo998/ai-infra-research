import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase76Experiments(unittest.TestCase):
    def _run_and_verify(self, exp_name: str) -> dict:
        exp_dir = ROOT / "experiments" / exp_name
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py)], check=True, cwd=str(ROOT))
        subprocess.run([sys.executable, str(run_py), "--check"], check=True, cwd=str(ROOT))
        return json.loads(summary_json.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_sm103_megamoe_kimi_sm120_tilelang(self) -> None:
        data = self._run_and_verify("flashinfer_sm103_nvfp4_tmem_megamoe_ep16_kimi_ess_sm120_tilelang")
        self.assertEqual(data["phase"], 76)
        self.assertEqual(data["priority"], 85)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["sm103_nvfp4_tmem_attention"]["tmem_ncols_pr"], 480)
        self.assertEqual(data["megamoe_ep16_relaxed_fence_dispatch"]["max_speedup_x"], 1.18)
        self.assertEqual(data["kimi_k3_tp12_ess_routing"]["persist_min_tokens_pr"], 961)
        self.assertEqual(data["sm120_fp8_varlen_and_device_alpha"]["rtx5090_max_speedup_x"], 1.404)

    def test_track_b_trtllm_glm52_rubin_vllm_fs_sglang_paged_experts(self) -> None:
        data = self._run_and_verify("trtllm_glm52_rubin_staircase_nvfp4_awq_vllm_fs_sglang_paged_experts")
        self.assertEqual(data["phase"], 76)
        self.assertEqual(data["priority"], 85)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["trtllm_glm52_rubin_and_nvfp4_awq"]["deepgemm_sm107_target"], "sm_100f")
        self.assertEqual(data["vllm_shared_fs_eexist_offload"]["fixed_eexist_failures"], 0)
        self.assertEqual(data["sglang_paged_experts_and_hicache_mamba"]["supported_format_count"], 12)

    def test_track_c_fa4_clc_packgqa_tma_verl_lora_pytorch_mesh(self) -> None:
        data = self._run_and_verify("fa4_sm100_clc_2cta_pack_gqa_ragged_tma_verl_lora_pytorch_mesh")
        self.assertEqual(data["phase"], 76)
        self.assertEqual(data["priority"], 85)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["fa4_sm100_clc_2cta_pack_gqa_tma"]["tma_h100_fixed_total_slots"], 888)
        self.assertEqual(data["verl_single_block_lora_staging"]["max_speedup_x"], 5.4)
        self.assertEqual(data["pytorch_devicemesh_xpu_mooncake"]["xpu_py_buildvalue_N_refcount"], 1)


if __name__ == "__main__":
    unittest.main()

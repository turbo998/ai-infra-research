from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase85Experiments(unittest.TestCase):
    def _run_and_load(self, exp_name: str) -> dict:
        exp_dir = ROOT / "experiments" / exp_name
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py)], check=True)
        subprocess.run([sys.executable, str(run_py), "--check"], check=True)
        return json.loads(summary_json.read_text(encoding="utf-8"))

    def test_flashinfer_sm10x_mxfp8_nvfp4_kimi_qsa_ag_norm_phase85(self) -> None:
        data = self._run_and_load("flashinfer_sm10x_mxfp8_nvfp4_kimi_qsa_ag_norm_phase85")
        self.assertEqual(data["phase"], 85)
        self.assertEqual(data["priority"], 94)
        self.assertEqual(data["cake_gated_act_mxfp8"]["small_tile"], [32, 32])
        self.assertEqual(data["cake_all_gather_matmul"]["final_pass_nsplit_tiles_2p"], 90)
        self.assertEqual(data["cake_mm_fp4"]["b200_quant_max_speedup"], 1.255)
        self.assertEqual(data["cake_kimi_k3_fused_router"]["gb300_t8192_bm16_speedup"], 1.80)
        self.assertEqual(data["cake_qsa_nvfp4_sm107"]["waves_on_r200"], 3)
        self.assertEqual(data["cake_warp_decode"]["new_ring_depth"], 5)
        self.assertEqual(data["cake_norm_combine_and_attn_res"]["attn_res_new_programs"], 181)

    def test_serving_pd_checksum_qsa_rocm_dbo_minimax_phase85(self) -> None:
        data = self._run_and_load("serving_pd_checksum_qsa_rocm_dbo_minimax_phase85")
        self.assertEqual(data["phase"], 85)
        self.assertTrue(data["sglang_pd_kv_checksum"]["radix_prefix_hit_causes_false_mismatch"])
        self.assertEqual(data["sglang_ltx2_vec_qknorm_rope"]["tx_reduction_factor"], 8)
        self.assertEqual(data["sglang_qsa_prefill_gather"]["new_triton_launches"], 1)
        self.assertEqual(data["sglang_eagle_hicache_restore"]["rematch_token_len"], 65)
        self.assertTrue(data["vllm_rocm_triton_ptr_range"]["uses_32bit_buffer_ops_after"])
        self.assertEqual(data["vllm_mrv2_dbo"]["captured_rows_after"], 128)
        self.assertEqual(data["vllm_mrv2_dbo"]["reported_ubatch_after"], 1)

    def test_fa4_2cta_verl_deepgemm_mamba_mooncake_nixl_phase85(self) -> None:
        data = self._run_and_load("fa4_2cta_verl_deepgemm_mamba_mooncake_nixl_phase85")
        self.assertEqual(data["phase"], 85)
        self.assertEqual(data["fa4_sm100_sm103"]["dqaccum_fp32_smem_bytes_saved"], 65536)
        self.assertEqual(data["fa4_sm100_sm103"]["sm103_capped_regs_other"], 128)
        self.assertEqual(data["verl_deepgemm_qat"]["mxfp4_kl_reduction_ratio"], 5.643)
        self.assertEqual(data["megatron_mamba_and_dp_cp"]["configured_timeout_sec"], 2700)
        self.assertEqual(data["mooncake_soft_pin"]["new_max_soft_pin_ttl_ms"], 604800000)
        self.assertEqual(data["nixl_dynamo_te"]["nixl_new_plugin_api_version"], 2)


if __name__ == "__main__":
    unittest.main()

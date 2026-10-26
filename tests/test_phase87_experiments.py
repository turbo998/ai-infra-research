from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase87Experiments(unittest.TestCase):
    def _run_and_verify(self, exp_name: str) -> dict:
        exp_dir = ROOT / "experiments" / exp_name
        run_py = exp_dir / "run.py"
        subprocess.run([sys.executable, str(run_py)], check=True, cwd=str(ROOT))
        subprocess.run([sys.executable, str(run_py), "--check"], check=True, cwd=str(ROOT))
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_bmm_nvfp4_hopper_msa_phase87(self) -> None:
        data = self._run_and_verify("flashinfer_bmm_nvfp4_hopper_msa_phase87")
        self.assertEqual(data["phase"], 87)
        self.assertEqual(data["bf16_bmm_swap_ab"]["routed_decode_shapes_per_arch"], 9)
        self.assertEqual(data["bf16_bmm_swap_ab"]["gb300_speedup_range_x"], [1.03, 1.22])
        self.assertEqual(data["cake_nvfp4_attention"]["sm120_r1_r7_speedup_range_x"], [1.41, 1.84])
        self.assertEqual(data["hopper_msa_and_host_guards"]["hopper_msa"]["head_files"], 68)
        self.assertEqual(
            data["hopper_msa_and_host_guards"]["minimax_h3_sm120_32bit_guard"]["max_safe_padded_tokens"],
            299520,
        )

    def test_sglang_vllm_trtllm_phase87(self) -> None:
        data = self._run_and_verify("sglang_vllm_trtllm_phase87")
        self.assertEqual(data["phase"], 87)
        self.assertEqual(data["sglang_mamba_dcp_grid"]["combined_lcm_align"], 192)
        self.assertEqual(data["sglang_mamba_dcp_grid"]["grid_aligned_dcp_donations_1024"], 4)
        self.assertEqual(data["sglang_k3_sm107_splitk"]["router_num_experts"], 896)
        self.assertEqual(data["vllm_and_trtllm_guards"]["trtllm_ulysses_packed_qkv"]["grid_dim_x"], 12288)

    def test_fa4_verl_mcore_mooncake_phase87(self) -> None:
        data = self._run_and_verify("fa4_verl_mcore_mooncake_phase87")
        self.assertEqual(data["phase"], 87)
        self.assertEqual(
            data["fa4_cudagraph_and_hd256"]["paged_kv_cudagraph_guard"]["host_syncs_during_capture_after"],
            0,
        )
        self.assertEqual(data["torchtitan_muse_spark_gb300"]["trainer_gpu_reduction_factor"], 2.0)
        self.assertEqual(data["torchtitan_muse_spark_gb300"]["expert_fsdp_group_reduction_factor"], 8.0)
        self.assertEqual(
            data["mooncake_and_lmcache"]["lmcache_packed_uint32"]["payload_reduction_factor_x"],
            9.0,
        )


if __name__ == "__main__":
    unittest.main()

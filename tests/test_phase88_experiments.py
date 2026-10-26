from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase88Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        exp_dir = ROOT / "experiments" / rel_dir
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py)], check=True, cwd=str(ROOT))
        subprocess.run([sys.executable, str(run_py), "--check"], check=True, cwd=str(ROOT))
        return json.loads(summary_json.read_text(encoding="utf-8"))

    def test_flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88(self) -> None:
        data = self._run_and_load("flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88")
        self.assertEqual(data["phase"], "Phase 88")
        self.assertEqual(data["priority"], 97)
        self.assertEqual(data["warp_decode"]["sm103a_changed_rows"], 124)
        self.assertEqual(data["warp_decode"]["launch_reduction_ratio"], 1.5)
        self.assertEqual(data["kda_cp_two_cta"]["max_b1_speedup_x"], 1.0754)
        self.assertEqual(data["kda_cp_two_cta"]["max_b3_speedup_x"], 1.0512)
        self.assertEqual(data["blackwell_bf16_fp4"]["small_m16_out_b200_speedup_x"], 1.205)
        self.assertEqual(data["grouped_mxfp8_quantize"]["gb200_k4096_16rows_speedup_x"], 1.3632)
        self.assertEqual(data["tilelang_sm120_and_stream"]["sm120_scale_load_reduction_x"], 6.0)

    def test_sglang_router_hicache_vllm_kv_offload_trtllm_phase88(self) -> None:
        data = self._run_and_load("sglang_router_hicache_vllm_kv_offload_trtllm_phase88")
        self.assertEqual(data["phase"], "Phase 88")
        self.assertEqual(data["priority"], 97)
        self.assertEqual(data["sglang_router_and_hicache"]["load_imbalance_reduction_x"], 8.0)
        self.assertEqual(data["sglang_dsv4_kda_ulysses"]["gfx950_kda_launch_reduction_x"], 5.0)
        self.assertEqual(data["vllm_rl_and_kv_offload"]["lru_prefix_retention_rate"], 1.0)
        self.assertEqual(data["vllm_rl_and_kv_offload"]["zmq_dealer_sndhwm"], 1000)
        self.assertEqual(data["trtllm_mamba_and_beam"]["tma_min_backing_bytes"], 131072)

    def test_fa4_sparse_mla_verl_mcore_mooncake_nixl_phase88(self) -> None:
        data = self._run_and_load("fa4_sparse_mla_verl_mcore_mooncake_nixl_phase88")
        self.assertEqual(data["phase"], "Phase 88")
        self.assertEqual(data["priority"], 97)
        self.assertEqual(
            data["fa4_sm100_sparse_and_cudagraph"][
                "sm100_empty_tile_nan_after_constexpr_zero_fill"
            ],
            0,
        )
        self.assertEqual(
            data["verl_megatron_te_training"]["te_grouped_wgrad_ulp_precision_gain_x"],
            65536,
        )
        self.assertEqual(data["mooncake_nixl_lmcache"]["nixl_39c6_small_p50_speedup_x"], 5.582)
        self.assertEqual(data["mooncake_nixl_lmcache"]["nixl_5b1a_small_p50_speedup_x"], 7.3582)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase89Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir: str) -> dict:
        script = ROOT / "experiments" / exp_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=ROOT)
        summary_path = ROOT / "experiments" / exp_dir / "summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_tilelang_phase89(self) -> None:
        data = self._run_and_load("flashinfer_nvfp4_mla_deepgemm_gdn_vsa_prims_sm107_tilelang_phase89")
        self.assertEqual(data["phase"], 89)
        self.assertEqual(data["priority"], 98)
        self.assertAlmostEqual(data["nvfp4_mla_decode"]["gb300_3split_speedup"], 1.109, places=3)
        self.assertAlmostEqual(data["batch_deepgemm_fp8"]["gb300_b128_m128_n512_k128_speedup"], 3.565, places=3)
        self.assertAlmostEqual(data["prims_ts_keeps"]["paged_dense_bf16_hq8_sq256_b64_gb300_speedup"], 5.58, places=2)
        self.assertAlmostEqual(data["prims_ts_mla_2cta"]["page16_b128_k8192_speedup"], 2.397, places=3)
        self.assertEqual(data["sm107_kimi_tilelang"]["sm107_supported_split_k"], [1, 2, 3, 4, 8])

    def test_track_b_sglang_vllm_trtllm_phase89(self) -> None:
        data = self._run_and_load("sglang_diffusion_vllm_topk_nvfp4_kv_offload_trtllm_grpc_phase89")
        self.assertEqual(data["phase"], 89)
        self.assertEqual(data["priority"], 98)
        self.assertEqual(data["sglang_diffusion_and_parser"]["cutedsl_norm_chunks_per_row"], 20)
        self.assertAlmostEqual(data["sglang_diffusion_and_parser"]["glm_streaming_scan_reduction_factor"], 500.5, places=1)
        self.assertEqual(data["vllm_topk_and_moe"]["persistent_topk_medium_hist_bytes"], 3072)
        self.assertEqual(data["trtllm_grpc_and_moe"]["default_protobuf_implementation"], "upb")

    def test_track_c_fa4_verl_mooncake_nixl_dynamo_phase89(self) -> None:
        data = self._run_and_load("fa4_sparse_mla_verl_mooncake_reshard_nixl_ep_dynamo_kvrouter_phase89")
        self.assertEqual(data["phase"], 89)
        self.assertEqual(data["priority"], 98)
        self.assertAlmostEqual(data["fa4_sm90_and_sm100"]["gb200_64head_64k_peak_mem_reduction_pct"], 66.75, places=2)
        self.assertEqual(data["verl_multinode_dp"]["gb200_validated_total_nodes"], 128)
        self.assertEqual(data["mooncake_reshard_and_store"]["crc32c_polynomial_hex"], "0x82F63B78")
        self.assertEqual(data["nixl_and_dynamo"]["nixl_ep_gb200_cpu_proxy_pass_rate_after"], "5/5")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Phase 82 (priority: 91) reproducible standard-library experiments."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.fa4_sm100_hd256_2cta_bwd_verl_veomni_bf16_rs_mooncake_batch_capi_nixl_dynamo_te.run import (
    run_experiment as run_track3,
)
from experiments.flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update.run import (
    run_experiment as run_track1,
)
from experiments.sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry.run import (
    run_experiment as run_track2,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase82Experiments(unittest.TestCase):
    def test_track1_flashinfer_warp_decode_rubin_mla_mok_ssm(self) -> None:
        summary = run_track1()
        self.assertEqual(summary["phase"], 82)
        self.assertEqual(summary["priority"], 91)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["warp_decode_k256_fc2"]["sm100a_routed_rows_total"], 38)
        self.assertEqual(summary["warp_decode_k256_fc2"]["sm103a_routed_rows_total"], 47)
        self.assertAlmostEqual(summary["warp_decode_k256_fc2"]["sm100_i256_t32_speedup_x"], 1.0505, places=4)
        self.assertEqual(summary["rubin_mla_nvfp4_decode"]["smem_delta_bytes"], 9216)
        self.assertAlmostEqual(summary["rubin_mla_nvfp4_decode"]["h448_q1_b1_kv64k_speedup_b_x"], 1.047, places=3)
        self.assertEqual(summary["mok_load_bins_and_ssm_update"]["ssm_shapes_total"], 122)
        self.assertAlmostEqual(summary["mok_load_bins_and_ssm_update"]["ssm_b200_speedup_max_x"], 1.253, places=3)
        self.assertAlmostEqual(summary["mok_load_bins_and_ssm_update"]["ssm_gb300_speedup_max_x"], 1.248, places=3)
        self.assertEqual(summary["mok_load_bins_and_ssm_update"]["ssd_device_files_after"], 72)

        script = (
            REPO_ROOT
            / "experiments"
            / "flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update"
            / "run.py"
        )
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=REPO_ROOT)

    def test_track2_sglang_diffusion_shm_cosmos3_vllm_kv_offload_trtllm(self) -> None:
        summary = run_track2()
        self.assertEqual(summary["phase"], 82)
        self.assertEqual(summary["priority"], 91)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["sglang_diffusion_cosmos3_dsv4"]["leaked_shm_mib_after"], 0)
        self.assertTrue(summary["sglang_diffusion_cosmos3_dsv4"]["decoder_conv3d_channels_last_3d"])
        self.assertFalse(summary["sglang_diffusion_cosmos3_dsv4"]["encoder_conv3d_channels_last_3d"])
        self.assertEqual(summary["vllm_kv_offload_and_trtllm_retry"]["skipped_chunks_after_fix"], 0)
        self.assertEqual(summary["vllm_kv_offload_and_trtllm_retry"]["trtllm_backoff_total_s"], 15)

        script = (
            REPO_ROOT
            / "experiments"
            / "sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry"
            / "run.py"
        )
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=REPO_ROOT)

    def test_track3_fa4_sm100_hd256_verl_veomni_mooncake_nixl_dynamo(self) -> None:
        summary = run_track3()
        self.assertEqual(summary["phase"], 82)
        self.assertEqual(summary["priority"], 91)
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["fa4_sm100_hd256_2cta_bwd"]["total_tmem_columns"], 512)
        self.assertEqual(summary["fa4_sm100_hd256_2cta_bwd"]["exchange_staging_bytes_after"], 0)
        self.assertAlmostEqual(summary["verl_veomni_and_te_grouped_tensor"]["rs_bandwidth_reduction_x"], 2.0, places=2)
        self.assertEqual(summary["mooncake_nixl_dynamo"]["mooncake_ffi_calls_after"], 1)
        self.assertTrue(summary["mooncake_nixl_dynamo"]["nixl_treq_roundtrip_exact"])
        self.assertTrue(summary["mooncake_nixl_dynamo"]["dynamo_monotonic_sequence_preserved"])

        script = (
            REPO_ROOT
            / "experiments"
            / "fa4_sm100_hd256_2cta_bwd_verl_veomni_bf16_rs_mooncake_batch_capi_nixl_dynamo_te"
            / "run.py"
        )
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()

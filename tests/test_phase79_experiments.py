from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.fa4_sm100_mla_kvhead_key_2cta_dq_reg_sm103_setmaxnreg_deepep_gin_nixl_obj import (
    run as exp_c,
)
from experiments.flashinfer_muse_spark_pdl_dcp_alltoall_grouped_mxfp8_qsa_sm107_minimax_tilelang import (
    run as exp_a,
)
from experiments.sglang_disagg_radix_checksum_ltx2_rope_vllm_mla_scale_skip_nccl_rl_trtllm import (
    run as exp_b,
)

ROOT = Path(__file__).resolve().parents[1]


class TestPhase79Experiments(unittest.TestCase):
    def test_track_a_flashinfer_and_tilelang(self) -> None:
        summary = exp_a.build_summary()
        exp_a.verify_summary(summary)
        self.assertEqual(summary["muse_spark_pdl"]["max_speedup_b200_x"], 1.509)
        self.assertEqual(summary["grouped_mxfp8_quantize"]["max_sparse_speedup_gb200_x"], 1.9357)
        self.assertEqual(summary["qsa_nvfp4_sm107_r200"]["splits_212_sms"], [33, 33, 26, 13, 6, 3, 1, 1])
        res = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "experiments"
                    / "flashinfer_muse_spark_pdl_dcp_alltoall_grouped_mxfp8_qsa_sm107_minimax_tilelang"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("OK:", res.stdout)

    def test_track_b_sglang_vllm_trtllm(self) -> None:
        summary = exp_b.build_summary()
        exp_b.verify_summary(summary)
        self.assertTrue(summary["sglang_disagg_radix_checksum"]["suffix_only_digest_match"])
        self.assertEqual(summary["sglang_ltx2_vectorized_rope"]["transaction_reduction_x"], 8.0)
        self.assertEqual(summary["trtllm_minimax_m3_piecewise"]["fixed_msa_out_cache_loc_count"], 19)
        res = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "experiments"
                    / "sglang_disagg_radix_checksum_ltx2_rope_vllm_mla_scale_skip_nccl_rl_trtllm"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("OK:", res.stdout)

    def test_track_c_fa4_deepep_deepgemm_nixl(self) -> None:
        summary = exp_c.build_summary()
        exp_c.verify_summary(summary)
        self.assertTrue(summary["fa4_mla_cache_key"]["unfixed_collision"])
        self.assertEqual(summary["fa4_sm103_setmaxnreg_cap"]["capped_num_regs_other"], 128)
        self.assertFalse(summary["nixl_fixes"]["ucx_err_mode_none_sets_err_handler_bit"])
        res = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "experiments"
                    / "fa4_sm100_mla_kvhead_key_2cta_dq_reg_sm103_setmaxnreg_deepep_gin_nixl_obj"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("OK:", res.stdout)


if __name__ == "__main__":
    unittest.main()

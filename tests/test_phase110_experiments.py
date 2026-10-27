from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_norm_combine_minimax_fp4_kimi_mla_phase110.run import (
    build_summary as build_track_a,
)
from experiments.sglang_swa_tombstone_disagg_kv_vllm_minimax_trtllm_phase110.run import (
    build_summary as build_track_b,
)
from experiments.te_qb_bounds_dist_weight_megatron_mfsdp_mooncake_nixl_phase110.run import (
    build_summary as build_track_c,
)

ROOT = Path(__file__).resolve().parents[1]


class TestPhase110Experiments(unittest.TestCase):
    def test_track_a_flashinfer_cutlass_tilelang(self) -> None:
        s = build_track_a()
        self.assertEqual(s["phase"], 110)
        self.assertEqual(s["priority"], 119)
        self.assertTrue(all(s["invariants"].values()))
        self.assertAlmostEqual(s["cake_fused_norm_combine_and_moe_ar"]["b200_64tok_speedup_x"], 1.0341, places=4)
        self.assertEqual(s["cutlass_and_tilelang"]["fixed_packed_offset_r3_c5"], 389)
        self.assertEqual(s["cutlass_and_tilelang"]["active_scan_lanes_with_offset"], 32)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "flashinfer_norm_combine_minimax_fp4_kimi_mla_phase110" / "run.py"),
                "--check",
            ],
            check=True,
        )

    def test_track_b_sglang_vllm_trtllm(self) -> None:
        s = build_track_b()
        self.assertEqual(s["phase"], 110)
        self.assertEqual(s["priority"], 119)
        self.assertTrue(all(s["invariants"].values()))
        self.assertEqual(s["sglang_swa_disagg_and_radix_gil"]["corrupted_protected_tokens_after"], 0)
        self.assertEqual(s["trtllm_adp_kv_and_sampler_logprobs"]["temp_vocab_buffer_mib_after"], 0.0)
        subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "experiments"
                    / "sglang_swa_tombstone_disagg_kv_vllm_minimax_trtllm_phase110"
                    / "run.py"
                ),
                "--check",
            ],
            check=True,
        )

    def test_track_c_te_megatron_torchtitan_mooncake_nixl(self) -> None:
        s = build_track_c()
        self.assertEqual(s["phase"], 110)
        self.assertEqual(s["priority"], 119)
        self.assertTrue(all(s["invariants"].values()))
        self.assertAlmostEqual(s["te_qb_bounds_and_dist_weight"]["prepare_saving_speedup_x"], 1.9231, places=4)
        self.assertEqual(s["mooncake_lmcache_dynamo_nixl"]["adjusted_prefill_with_fetch_credit"], 70.0)
        subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "experiments"
                    / "te_qb_bounds_dist_weight_megatron_mfsdp_mooncake_nixl_phase110"
                    / "run.py"
                ),
                "--check",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()

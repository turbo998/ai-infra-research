from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase77Experiments(unittest.TestCase):
    def _run_and_load(self, exp_name: str) -> dict:
        script = ROOT / "experiments" / exp_name / "run.py"
        summary_path = ROOT / "experiments" / exp_name / "summary.json"
        res = subprocess.run(
            [sys.executable, str(script), "--output", str(summary_path), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
        import json

        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_mxfp8_ep8_warp_decode_tilelang(self) -> None:
        data = self._run_and_load(
            "flashinfer_gated_act_mxfp8_ep8_megamoe_warp_decode_norm_combine_tilelang"
        )
        self.assertEqual(data["phase"], 77)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["cake_gated_act_mxfp8"]["max_r200_speedup"], 1.697)
        self.assertEqual(data["cake_gated_act_mxfp8"]["max_b200_speedup"], 1.501)
        self.assertEqual(data["ep8_megamoe_arrive_mailbox"]["max_cg_speedup"], 1.08)
        self.assertEqual(
            data["warp_decode_router_norm_and_tilelang"]["router_gemm_max_speedup_sm103"],
            1.733,
        )

    def test_track_b_sglang_hisparse_dp_attn_ipc_vllm_trtllm(self) -> None:
        data = self._run_and_load(
            "sglang_hisparse_dsv4_dp_attn_bucket_diffusion_ipc_vllm_spec_autofit_kv_reset_trtllm"
        )
        self.assertEqual(data["phase"], 77)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["hisparse_dsv4_backup"]["corrupted_tokens_legacy"], 32)
        self.assertEqual(data["hisparse_dsv4_backup"]["corrupted_tokens_fixed"], 0)
        self.assertEqual(data["dp_attention_alignment"]["fixed_target_bucket"], 24)
        self.assertEqual(data["diffusion_ipc_cleanup"]["fixed_total_leaked_mib"], 0)

    def test_track_c_fa4_tma_nan_guard_verl_topk_cp_mooncake_mlu(self) -> None:
        data = self._run_and_load(
            "fa4_sm90_sm100_seqlenk_tma_nan_guard_verl_topk_cp_pack_mooncake_mlu"
        )
        self.assertEqual(data["phase"], 77)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(
            data["fa4_5d_ragged_tma_nan_guard"]["ragged_rows_limit"], 33554432
        )
        self.assertEqual(
            data["verl_topk_distill_alignment"]["rank_results"][0][
                "legacy_misaligned_valid_tokens"
            ],
            128,
        )
        self.assertEqual(
            data["verl_topk_distill_alignment"]["rank_results"][1][
                "pr8332_reported_misaligned_valid_tokens"
            ],
            136,
        )
        self.assertEqual(
            data["verl_topk_distill_alignment"]["rank_results"][1][
                "legacy_misaligned_pct"
            ],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()

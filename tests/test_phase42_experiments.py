import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_rubin_nvfp4_mla_mnnvl_handle.run import (
    build_summary as build_rubin_summary,
)
from experiments.megatron_lsmuon_stream_mooncake_acc_quota_nixl.run import (
    build_summary as build_lsmuon_summary,
)
from experiments.sglang_pd_dp_role_switch_vllm_dsv41_replay.run import (
    build_summary as build_sglang_summary,
)

ROOT = Path(__file__).resolve().parents[1]


class TestPhase42Experiments(unittest.TestCase):
    def test_flashinfer_rubin_nvfp4_mla_mnnvl_handle(self) -> None:
        summary = build_rubin_summary()
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["rubin_mla_nvfp4_decode"]["nvfp4_kv_bytes_per_token"], 352)
        self.assertEqual(summary["rubin_mla_nvfp4_decode"]["peak_speedup_h96_q4_b64_kv128k"], 1.248)
        self.assertEqual(
            summary["mnnvl_collective_handle"]["b200_single_host_no_imex_fallback"]["final_handle_type"],
            "POSIX_FD",
        )

    def test_sglang_pd_dp_role_switch_vllm_dsv41_replay(self) -> None:
        summary = build_sglang_summary()
        self.assertTrue(all(summary["invariants"].values()))
        self.assertFalse(
            summary["sglang_pd_dp_role_switch"]["two_phase_prepare_partial_failure"]["split_brain"]
        )
        self.assertEqual(
            summary["vllm_dsv41_replay_rows"]["fixed_target_scoring"]["replayed_rows"], 16
        )

    def test_megatron_lsmuon_stream_mooncake_acc_quota_nixl(self) -> None:
        summary = build_lsmuon_summary()
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(
            summary["lsmuon_streaming_and_te_alignment"]["peak_memory_reduction_ratio"], 16.0
        )
        self.assertEqual(
            summary["mooncake_tenant_quota"]["owed_during_upsert_processing"], 900
        )
        self.assertEqual(
            summary["nixl_xfer_complete_trace"]["fixed_complete_markers"], 50
        )

    def test_phase41_cli_check_mode(self) -> None:
        scripts = [
            ROOT / "experiments" / "flashinfer_rubin_nvfp4_mla_mnnvl_handle" / "run.py",
            ROOT / "experiments" / "sglang_pd_dp_role_switch_vllm_dsv41_replay" / "run.py",
            ROOT / "experiments" / "megatron_lsmuon_stream_mooncake_acc_quota_nixl" / "run.py",
        ]
        for script in scripts:
            proc = subprocess.run(
                [sys.executable, str(script), "--check"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{script} failed: {proc.stderr or proc.stdout}")


if __name__ == "__main__":
    unittest.main()

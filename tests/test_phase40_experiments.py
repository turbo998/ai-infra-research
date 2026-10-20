import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_cake_grouped_mxfp8_vllm_dsv4_eplb_kimi.run import (
    build_summary as build_cake_summary,
)
from experiments.megatron_mfsdp_v2_muon_offline_kd_nixl_prometheus.run import (
    build_summary as build_mfsdp_summary,
)
from experiments.sglang_detok_batch_radix_child_count_dynamo_coredump.run import (
    build_summary as build_sglang_summary,
)


ROOT = Path(__file__).resolve().parents[1]


class TestPhase40Experiments(unittest.TestCase):
    def test_flashinfer_cake_grouped_mxfp8_vllm_dsv4_eplb_kimi(self) -> None:
        summary = build_cake_summary()
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["author_benchmarks"][1]["speedup"], 0.992)
        self.assertEqual(summary["vllm_dsv4_eplb_and_kimi_k3"]["fixed_bad_redundant_slots"], 0)
        self.assertEqual(
            summary["vllm_dsv4_eplb_and_kimi_k3"]["kimi_k3_gb200_agentx_profile"]["agentx_topology"],
            "TP8xPP2+DCP8",
        )

    def test_sglang_detok_batch_radix_child_count_dynamo_coredump(self) -> None:
        summary = build_sglang_summary()
        self.assertTrue(all(summary["invariants"].values()))
        self.assertEqual(summary["detokenizer_batching"]["zmq_send_reduction_ratio"], 4.0)
        self.assertEqual(summary["radix_collect_path_info"]["chain_after_branch_children"], [2, 0])
        self.assertEqual(summary["dynamo_cuda_coredump_wait"]["default_disabled"]["effective_wait_secs"], 0.0)

    def test_megatron_mfsdp_v2_muon_offline_kd_nixl_prometheus(self) -> None:
        summary = build_mfsdp_summary()
        self.assertTrue(all(summary["invariants"].values()))
        self.assertTrue(all(c["max_abs_diff_vs_unsharded"] == 0.0 for c in summary["mfsdp_v2_muon_cases"]))
        self.assertEqual(summary["offline_kd_v2"]["bit17_storage_savings_pct"], 87.5)
        self.assertTrue(summary["nixl_prometheus_mp"]["same_dlopen_handle"])

    def test_phase40_cli_check_mode(self) -> None:
        scripts = [
            ROOT / "experiments" / "flashinfer_cake_grouped_mxfp8_vllm_dsv4_eplb_kimi" / "run.py",
            ROOT / "experiments" / "sglang_detok_batch_radix_child_count_dynamo_coredump" / "run.py",
            ROOT / "experiments" / "megatron_mfsdp_v2_muon_offline_kd_nixl_prometheus" / "run.py",
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

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase29Experiments(unittest.TestCase):
    def test_sglang_kv_shard_pd_transfer(self) -> None:
        script = ROOT / "experiments" / "sglang_kv_shard_pd_transfer" / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] sglang_kv_shard_pd_transfer", res.stdout)
        summary = json.loads(
            (script.parent / "results" / "summary.json").read_text(encoding="utf-8")
        )
        m = summary["summary_metrics"]
        self.assertTrue(m["rotated_chain_exact_reconstruction"])
        self.assertEqual(m["shard2_bootstrap_limit_tokens"], 32576)
        self.assertEqual(m["shard4_bootstrap_limit_tokens"], 65216)
        self.assertEqual(m["shard8_bootstrap_limit_tokens"], 130496)
        self.assertEqual(m["needle_prompt_to_single_rank_ratio"], 1.5)
        self.assertEqual(m["mla_tp4_required_prefill_responses"], 4)
        self.assertTrue(m["dcp_conflict_rejected"])

    def test_pytorch_cuda_cow_nccl_lazy_mempool(self) -> None:
        script = ROOT / "experiments" / "pytorch_cuda_cow_nccl_lazy_mempool" / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] pytorch_cuda_cow_nccl_lazy_mempool", res.stdout)
        summary = json.loads(
            (script.parent / "results" / "summary.json").read_text(encoding="utf-8")
        )
        m = summary["summary_metrics"]
        self.assertEqual(m["legacy_stale_replay_reads"], 8)
        self.assertEqual(m["patched_stale_replay_reads"], 0)
        self.assertEqual(m["uncompleted_peak_memory_mib"], 512.0)
        self.assertEqual(m["completed_steal_peak_memory_mib"], 256.0)
        self.assertEqual(m["memory_reduction_pct"], 50.0)
        self.assertEqual(m["uncompleted_steal_stall_ms"], 45.0)
        self.assertEqual(m["completed_steal_stall_ms"], 10.0)
        self.assertEqual(m["stall_reduction_pct"], 77.78)
        self.assertEqual(m["rejected_illegal_capture_crossings"], 3)
        self.assertEqual(m["nccl_lazy_active_pools_after_deregister"], 0)

    def test_nixl_ucx_vram_hint_triton_cdna5_barrier(self) -> None:
        script = (
            ROOT / "experiments" / "nixl_ucx_vram_hint_triton_cdna5_barrier" / "run.py"
        )
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] nixl_ucx_vram_hint_triton_cdna5_barrier", res.stdout)
        summary = json.loads(
            (script.parent / "results" / "summary.json").read_text(encoding="utf-8")
        )
        m = summary["summary_metrics"]
        self.assertEqual(m["ze_only_auto_hint"], "ze-device")
        self.assertEqual(m["mixed_ze_cuda_auto_hint"], "unknown")
        self.assertEqual(m["unpatched_stuck_empty_notifs"], 2)
        self.assertEqual(m["patched_stuck_empty_notifs"], 0)
        self.assertEqual(m["unpatched_libfabric_vmm_failures"], 3)
        self.assertEqual(m["patched_libfabric_vmm_failures"], 0)
        self.assertEqual(m["unpatched_cdna5_deadlocked_waves"], 14)
        self.assertEqual(m["patched_cdna5_deadlocked_waves"], 0)


if __name__ == "__main__":
    unittest.main()

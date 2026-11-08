import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGpuClusterFaultRecoverySelection(unittest.TestCase):
    def test_results_match_and_invariants_hold(self):
        repo_root = Path(__file__).resolve().parent.parent
        run_script = repo_root / "experiments" / "gpu_cluster_fault_recovery_selection" / "run.py"
        results_path = repo_root / "experiments" / "gpu_cluster_fault_recovery_selection" / "results.json"

        subprocess.run([sys.executable, str(run_script), "--check"], check=True)
        data = json.loads(results_path.read_text(encoding="utf-8"))

        self.assertEqual(data["provenance"]["independently_reproduced"], False)
        self.assertEqual(data["provenance"]["gpu_cluster_hardware_measured"], "not_measured")

        archs = {a["id"]: a for a in data["architectures"]}
        t1 = archs["traditional_stop_the_world_restart"]
        t2 = archs["fast_hang_detect_warm_spare_swap"]
        t3 = archs["inplace_comm_shrink_elastic_async_ckpt"]
        t4 = archs["torchft_in_memory_replica_peer_recovery"]

        self.assertEqual(t1["recovery_outage_sec"], 2520.0)
        self.assertEqual(t1["total_lost_sec_per_fault"], 3420.0)
        self.assertEqual(t1["effective_goodput_pct"], 84.17)

        self.assertEqual(t2["recovery_outage_sec"], 330.0)
        self.assertEqual(t2["total_lost_sec_per_fault"], 1230.0)
        self.assertEqual(t2["effective_goodput_pct"], 94.31)

        self.assertEqual(t3["recovery_outage_sec"], 65.0)
        self.assertEqual(t3["total_lost_sec_per_fault"], 215.0)
        self.assertEqual(t3["effective_goodput_pct"], 99.0)

        self.assertEqual(t4["recovery_outage_sec"], 28.0)
        self.assertEqual(t4["total_lost_sec_per_fault"], 28.0)
        self.assertEqual(t4["effective_goodput_pct"], 99.87)
        self.assertGreater(t4["monthly_savings_vs_baseline_usd"], 290000.0)


if __name__ == "__main__":
    unittest.main()

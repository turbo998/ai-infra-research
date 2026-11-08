from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.agentic_sandbox_gpu_coldstart_selection.run import RESULTS_PATH, compute_model


class TestAgenticSandboxGpuColdstartSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_coldstart_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_k8s_cold_oci_pull"]
        t2 = tiers["tier2_lazy_oci_soci_mmap"]
        t3 = tiers["tier3_firecracker_microvm_snapshot"]
        t4 = tiers["tier4_criu_cuda_checkpoint_rdma"]

        self.assertEqual(t1["p50_coldstart_ms"], 25500.0)
        self.assertFalse(t1["sla_500ms_pass"])
        self.assertEqual(t2["p50_coldstart_ms"], 5200.0)
        self.assertAlmostEqual(t2["coldstart_speedup_x"], 4.90, places=2)
        self.assertEqual(t3["p50_coldstart_ms"], 125.0)
        self.assertAlmostEqual(t3["coldstart_speedup_x"], 204.0, places=1)
        self.assertTrue(t3["sla_500ms_pass"])
        self.assertEqual(t4["p50_coldstart_ms"], 420.0)
        self.assertAlmostEqual(t4["coldstart_speedup_x"], 60.71, places=2)
        self.assertTrue(t4["sla_500ms_pass"])
        self.assertEqual(t4["cost_savings_pct"], 77.3)


if __name__ == "__main__":
    unittest.main()

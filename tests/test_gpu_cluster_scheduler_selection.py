from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.gpu_cluster_scheduler_selection.run import RESULTS_PATH, compute_model


class TestGpuClusterSchedulerSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_scheduler_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_vanilla_k8s_default"]
        t2 = tiers["tier2_standalone_slurm_hpc"]
        t3 = tiers["tier3_k8s_volcano_kueue_gang"]
        t4 = tiers["tier4_kueue_tas_k8s_dra"]

        self.assertEqual(t1["deadlock_fragmentation_pct"], 18.5)
        self.assertEqual(t1["effective_goodput_pct"], 48.2)
        self.assertEqual(t2["effective_goodput_pct"], 74.0)
        self.assertEqual(t3["effective_goodput_pct"], 71.5)
        self.assertEqual(t4["comm_overhead_pct"], 3.2)
        self.assertEqual(t4["effective_goodput_pct"], 88.6)
        self.assertEqual(t4["monthly_recovered_value_vs_tier1_usd"], 377497.60)
        self.assertEqual(t4["cost_savings_pct"], 45.6)


if __name__ == "__main__":
    unittest.main()

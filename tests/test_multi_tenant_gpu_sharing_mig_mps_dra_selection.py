from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.multi_tenant_gpu_sharing_mig_mps_dra_selection.run import (
    evaluate_gpu_sharing_tiers,
)


class TestMultiTenantGpuSharingMigMpsDraSelection(unittest.TestCase):
    def test_provenance_and_sharing_invariants(self) -> None:
        data = evaluate_gpu_sharing_tiers()
        self.assertEqual(
            data["experiment_id"], "multi_tenant_gpu_sharing_mig_mps_dra_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier2_vs_tier1_throughput_gain_x"], 2.35)
        self.assertEqual(summary["tier2_fault_blast_radius_pct"], 100.0)
        self.assertEqual(summary["tier3_fault_blast_radius_pct"], 0.0)
        self.assertEqual(summary["tier4_gpu_utilization_pct"], 88.5)
        self.assertEqual(summary["tier4_vs_dedicated_gpu_saving_pct"], 71.4)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "multi_tenant_gpu_sharing_mig_mps_dra_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_gpu_sharing_tiers())


if __name__ == "__main__":
    unittest.main()

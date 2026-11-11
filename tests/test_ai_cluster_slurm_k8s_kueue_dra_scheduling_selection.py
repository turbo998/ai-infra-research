"""Unit tests for Round 77 AI Cluster Scheduler (Slurm vs Vanilla K8s vs Kueue/Volcano vs Kueue TAS + K8s DRA + Sunk) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_cluster_slurm_k8s_kueue_dra_scheduling_selection.run import (
    evaluate_ai_cluster_slurm_k8s_kueue_dra_scheduling_matrix,
)


class TestAiClusterSlurmK8sKueueDraSchedulingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_cluster_slurm_k8s_kueue_dra_scheduling_matrix()
        self.assertEqual(
            data["experiment_id"],
            "ai_cluster_slurm_k8s_kueue_dra_scheduling_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_unified_k8s_kueue_tas_plus_dra_plus_sunk_slurm_bridge"][
                "overall_24x7_cluster_utilization_pct"
            ],
            93.0,
        )
        self.assertEqual(
            tiers["tier4_unified_k8s_kueue_tas_plus_dra_plus_sunk_slurm_bridge"][
                "subnode_gpu_nic_pcie_alignment_pct"
            ],
            100.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_utilization_gain_pts"],
            20.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_cluster_slurm_k8s_kueue_dra_scheduling_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_cluster_slurm_k8s_kueue_dra_scheduling_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

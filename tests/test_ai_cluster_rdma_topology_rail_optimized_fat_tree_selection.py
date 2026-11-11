"""Unit tests for Round 75 4,096-GPU AI Cluster RDMA Network Topology (Rail-Optimized + PXN vs Fat-Tree) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_cluster_rdma_topology_rail_optimized_fat_tree_selection.run import (
    evaluate_ai_cluster_rdma_topology_rail_optimized_fat_tree_matrix,
)


class TestAiClusterRdmaTopologyRailOptimizedFatTreeSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_cluster_rdma_topology_rail_optimized_fat_tree_matrix()
        self.assertEqual(
            data["experiment_id"],
            "ai_cluster_rdma_topology_rail_optimized_fat_tree_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier3_rail_optimized_fat_tree_plus_nccl_pxn_nvswitch"]["effective_node_collective_bw_gbps"],
            360.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_capex_savings_usd"],
            7000000.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_power_reduction_pct"],
            40.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_cluster_rdma_topology_rail_optimized_fat_tree_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_cluster_rdma_topology_rail_optimized_fat_tree_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

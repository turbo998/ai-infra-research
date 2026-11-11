"""Unit tests for Round 72 Cluster Model Weight Loading, Nydus/SOCI Lazy OCI & RDMA P2P Broadcast Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.model_weight_loading_oci_safetensors_rdma_broadcast_selection.run import (
    evaluate_model_weight_loading_oci_safetensors_rdma_broadcast_matrix,
)


class TestModelWeightLoadingOciSafetensorsRdmaBroadcastSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_model_weight_loading_oci_safetensors_rdma_broadcast_matrix()
        self.assertEqual(
            data["experiment_id"],
            "model_weight_loading_oci_safetensors_rdma_broadcast_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier4_nydus_plus_leader_once_read_rdma_p2p_ring_broadcast_pinned_cache"][
                "cluster_64node_rollout_time_s"
            ],
            10.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_rollout_speedup_x"],
            50.0,
        )
        self.assertEqual(
            data["summary"]["tier4_vs_tier1_storage_traffic_reduction_x"],
            64.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "model_weight_loading_oci_safetensors_rdma_broadcast_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_model_weight_loading_oci_safetensors_rdma_broadcast_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for AI Cloud-Native Object Storage High-Throughput Streaming Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_object_storage_s3_express_mountpoint_rdma_selection.run import (
    evaluate_ai_object_storage_s3_express_mountpoint_rdma_matrix,
)


class TestAiObjectStorageS3ExpressMountpointRdmaSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_object_storage_s3_express_mountpoint_rdma_matrix()
        self.assertEqual(
            data["experiment_id"],
            "ai_object_storage_s3_express_mountpoint_rdma_selection",
        )
        self.assertEqual(data["round"], 183)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier3_native_rust_crt_parallel_range_s5cmd"]["node_read_throughput_gbs"],
            25.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_throughput_speedup_ratio"],
            240.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_ttfb_reduction_ratio"],
            20.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_object_storage_s3_express_mountpoint_rdma_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_object_storage_s3_express_mountpoint_rdma_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

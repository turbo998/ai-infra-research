"""Unit tests for Round 73 Billion-Scale Vector Database (HNSW vs IVF-PQ vs DiskANN vs GPU CAGRA + RaBitQ) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.vector_database_billion_scale_diskann_cagra_selection.run import (
    evaluate_vector_database_billion_scale_diskann_cagra_matrix,
)


class TestVectorDatabaseBillionScaleDiskannCagraSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_vector_database_billion_scale_diskann_cagra_matrix()
        self.assertEqual(
            data["experiment_id"],
            "vector_database_billion_scale_diskann_cagra_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_gpu_cuvs_cagra_plus_rabitq_and_nvme_hybrid"]["recall_at_10_pct"],
            96.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_dram_reduction_ratio"],
            20.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_build_speedup_x"],
            12.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "vector_database_billion_scale_diskann_cagra_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_vector_database_billion_scale_diskann_cagra_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

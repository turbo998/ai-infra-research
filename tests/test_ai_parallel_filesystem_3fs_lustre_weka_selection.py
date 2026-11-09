from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.ai_parallel_filesystem_3fs_lustre_weka_selection.run import (
    evaluate_parallel_filesystem_tiers,
)


class TestAiParallelFilesystem3fsLustreWekaSelection(unittest.TestCase):
    def test_provenance_and_storage_invariants(self) -> None:
        data = evaluate_parallel_filesystem_tiers()
        self.assertEqual(
            data["experiment_id"], "ai_parallel_filesystem_3fs_lustre_weka_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier1_throughput_gain_x"], 26.29)
        self.assertEqual(summary["tier4_vs_tier2_random_iops_gain_x"], 7.33)
        self.assertEqual(summary["tier4_10tb_checkpoint_load_sec"], 1.5)
        self.assertEqual(summary["tier3_vs_tier4_3yr_tco_savings_usd"], 1480000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "ai_parallel_filesystem_3fs_lustre_weka_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_parallel_filesystem_tiers())


if __name__ == "__main__":
    unittest.main()

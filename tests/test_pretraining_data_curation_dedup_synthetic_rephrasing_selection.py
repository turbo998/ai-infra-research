"""Unit tests for Round 58 Trillion-Token Pretraining Data Curation, Deduplication & Synthetic Rephrasing Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.pretraining_data_curation_dedup_synthetic_rephrasing_selection.run import (
    evaluate_pretraining_data_curation_dedup_matrix,
)


class TestPretrainingDataCurationDedupSyntheticRephrasingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_pretraining_data_curation_dedup_matrix()
        self.assertEqual(
            data["experiment_id"],
            "pretraining_data_curation_dedup_synthetic_rephrasing_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_gpu_curator_plus_nemotron_cc_synthetic_rephrasing"]["downstream_8b_benchmark_score_pct"],
            78.0,
        )
        self.assertLess(
            tiers["tier3_gpu_nemo_curator_cudf_fineweb_edu_classifier"]["curation_wallclock_hours"],
            40.0,
        )
        self.assertGreater(
            data["summary"]["tier3_vs_tier2_dedup_speedup_ratio"],
            15.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "pretraining_data_curation_dedup_synthetic_rephrasing_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_pretraining_data_curation_dedup_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

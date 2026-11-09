from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.agentic_memory_state_persistence_selection.run import (
    evaluate_agentic_memory_tiers,
)


class TestAgenticMemoryStatePersistenceSelection(unittest.TestCase):
    def test_provenance_and_memory_invariants(self) -> None:
        data = evaluate_agentic_memory_tiers()
        self.assertEqual(
            data["experiment_id"], "agentic_memory_state_persistence_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier3_temporal_update_gain_pt"], 36.3)
        self.assertEqual(summary["tier4_vs_tier1_cost_saving_pct"], 98.8)
        self.assertEqual(summary["tier4_vs_tier1_ttft_speedup_x"], 12.67)
        self.assertEqual(summary["tier4_overall_longmemeval_accuracy_pct"], 93.2)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "agentic_memory_state_persistence_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_agentic_memory_tiers())


if __name__ == "__main__":
    unittest.main()

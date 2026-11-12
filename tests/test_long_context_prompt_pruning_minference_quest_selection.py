from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.long_context_prompt_pruning_minference_quest_selection.run import (
    evaluate_long_context_minference_quest_tiers,
)


class TestLongContextPromptPruningMinferenceQuestSelection(unittest.TestCase):
    def test_provenance_and_minference_quest_invariants(self) -> None:
        data = evaluate_long_context_minference_quest_tiers()
        self.assertEqual(
            data["experiment_id"],
            "long_context_prompt_pruning_minference_quest_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier1_ttft_speedup_x"], 8.13)
        self.assertEqual(summary["tier4_vs_tier1_tpot_speedup_x"], 4.74)
        self.assertEqual(summary["tier4_vs_tier3_multi_turn_recall_gain_pt"], 34.1)
        self.assertEqual(summary["tier4_vs_tier1_throughput_gain_x"], 7.13)
        self.assertEqual(summary["tier4_vs_tier1_annual_savings_usd"], 1517300)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "long_context_prompt_pruning_minference_quest_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_long_context_minference_quest_tiers())


if __name__ == "__main__":
    unittest.main()

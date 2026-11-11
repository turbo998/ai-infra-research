"""Unit tests for Round 69 VLM High-Resolution AnyRes & Visual Token Compression Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.vlm_anyres_visual_token_compression_pruning_selection.run import (
    evaluate_vlm_anyres_visual_token_compression_matrix,
)


class TestVlmAnyresVisualTokenCompressionPruningSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_vlm_anyres_visual_token_compression_matrix()
        self.assertEqual(
            data["experiment_id"],
            "vlm_anyres_visual_token_compression_pruning_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_navit_pixel_shuffle_plus_text_guided_pruning_fastv_visionzip"]["fine_grained_vqa_accuracy_pct"],
            89.0,
        )
        self.assertLess(
            tiers["tier4_navit_pixel_shuffle_plus_text_guided_pruning_fastv_visionzip"]["llm_visual_tokens_per_image"],
            800,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_token_compression_ratio"],
            12.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "vlm_anyres_visual_token_compression_pruning_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_vlm_anyres_visual_token_compression_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

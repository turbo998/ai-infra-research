"""Unit tests for Round 369 Massive Offline Document Batch Processing & Knowledge Extraction Throughput Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.massive_document_batch_extraction_async_spot_prefix_selection.run import (
    evaluate_massive_document_batch_extraction_async_spot_prefix,
)


class TestMassiveDocumentBatchExtractionAsyncSpotPrefixSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_massive_document_batch_extraction_async_spot_prefix()
        self.assertEqual(
            data["experiment_id"],
            "massive_document_batch_extraction_async_spot_prefix_selection",
        )
        self.assertEqual(data["round"], 369)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_2d_prefix_length_sorted_bucketing_plus_offpeak_backfill"]["effective_throughput_docs_s"],
            tiers["tier1_naive_multithreaded_sync_online_api_polling"]["effective_throughput_docs_s"],
        )
        self.assertGreater(
            data["summary"]["tier4_cost_savings_vs_tier1_online_pct"],
            90.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "massive_document_batch_extraction_async_spot_prefix_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_massive_document_batch_extraction_async_spot_prefix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

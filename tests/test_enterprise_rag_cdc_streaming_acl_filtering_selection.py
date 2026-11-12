from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.enterprise_rag_cdc_streaming_acl_filtering_selection.run import (
    evaluate_enterprise_rag_cdc_acl_filtering_tiers,
)


class TestEnterpriseRagCdcStreamingAclFilteringSelection(unittest.TestCase):
    def test_provenance_and_rag_cdc_acl_invariants(self) -> None:
        data = evaluate_enterprise_rag_cdc_acl_filtering_tiers()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_rag_cdc_streaming_acl_filtering_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier2_recall_gain_pt"], 71.8)
        self.assertEqual(summary["tier4_vs_tier3_recall_gain_pt"], 28.0)
        self.assertEqual(summary["tier4_vs_tier3_latency_speedup_x"], 9.41)
        self.assertEqual(summary["tier4_vs_tier1_embedding_cost_reduction_pct"], 98.8)
        self.assertEqual(summary["tier4_vs_tier1_annual_savings_usd"], 794000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "enterprise_rag_cdc_streaming_acl_filtering_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_enterprise_rag_cdc_acl_filtering_tiers())


if __name__ == "__main__":
    unittest.main()

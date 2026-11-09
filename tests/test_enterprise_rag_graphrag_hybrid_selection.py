from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.enterprise_rag_graphrag_hybrid_selection.run import (
    evaluate_enterprise_rag_tiers,
)


class TestEnterpriseRagGraphragHybridSelection(unittest.TestCase):
    def test_provenance_and_rag_invariants(self) -> None:
        data = evaluate_enterprise_rag_tiers()
        self.assertEqual(
            data["experiment_id"], "enterprise_rag_graphrag_hybrid_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier2_global_multihop_gain_pt"], 50.0)
        self.assertEqual(summary["tier4_vs_eager_graphrag_index_cost_saving_pct"], 99.2)
        self.assertEqual(summary["tier3_single_hop_factoid_accuracy_pct"], 94.6)
        self.assertEqual(summary["tier4_query_p95_latency_ms"], 680.0)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "enterprise_rag_graphrag_hybrid_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_enterprise_rag_tiers())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Enterprise Production RAG Architecture, Pre-Filter ACL & Selective Hybrid GraphRAG Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_production_rag_architecture_acl_cdc_graphrag_selection.run import (
    evaluate_enterprise_production_rag_architecture_acl_cdc_graphrag,
)


class TestEnterpriseProductionRagArchitectureAclCdcGraphragSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_production_rag_architecture_acl_cdc_graphrag()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_production_rag_architecture_acl_cdc_graphrag_selection",
        )
        self.assertEqual(data["round"], 374)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertEqual(
            tiers["tier4_adaptive_hybrid_rag_plus_selective_lazy_graphrag_router"]["acl_starvation_empty_rate_pct"],
            0.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_graphrag_tco_savings_pct"],
            80.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_production_rag_architecture_acl_cdc_graphrag_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_production_rag_architecture_acl_cdc_graphrag()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

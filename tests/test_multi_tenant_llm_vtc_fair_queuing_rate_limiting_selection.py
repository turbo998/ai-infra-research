"""Unit tests for Round 74 Multi-Tenant LLM Fair Queuing (VTC Virtual Token Counter) & Work-Conserving Rate Limiting Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.multi_tenant_llm_vtc_fair_queuing_rate_limiting_selection.run import (
    evaluate_multi_tenant_llm_vtc_fair_queuing_rate_limiting_matrix,
)


class TestMultiTenantLlmVtcFairQueuingRateLimitingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_multi_tenant_llm_vtc_fair_queuing_rate_limiting_matrix()
        self.assertEqual(
            data["experiment_id"],
            "multi_tenant_llm_vtc_fair_queuing_rate_limiting_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreaterEqual(
            tiers["tier4_slo_hierarchical_vtc_work_conserving_borrow_chunk_preempt"]["jains_fairness_index"],
            0.98,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_ttft_speedup_x"],
            20.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier3_utilization_gain_pts"],
            20.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "multi_tenant_llm_vtc_fair_queuing_rate_limiting_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_multi_tenant_llm_vtc_fair_queuing_rate_limiting_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

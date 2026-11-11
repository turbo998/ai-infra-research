"""Unit tests for Round 65 Enterprise Multi-Agent Orchestration, StateGraph DAG & Typed Blackboard Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.multi_agent_orchestration_dag_blackboard_handoff_selection.run import (
    evaluate_multi_agent_orchestration_dag_matrix,
)


class TestMultiAgentOrchestrationDagBlackboardHandoffSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_multi_agent_orchestration_dag_matrix()
        self.assertEqual(
            data["experiment_id"],
            "multi_agent_orchestration_dag_blackboard_handoff_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_parallel_dag_scoped_blackboard_direct_handoff"]["workflow_completion_accuracy_pct"],
            92.0,
        )
        self.assertEqual(
            tiers["tier4_parallel_dag_scoped_blackboard_direct_handoff"]["infinite_loop_timeout_rate_pct"],
            0.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_token_reduction_ratio"],
            9.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "multi_agent_orchestration_dag_blackboard_handoff_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_multi_agent_orchestration_dag_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

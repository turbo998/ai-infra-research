"""Unit tests for Enterprise Multi-Agent Production Serving, MCP Tool Gateway & Planner-Executor Routing Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_agentic_workflow_durable_state_mcp_routing_selection.run import (
    evaluate_enterprise_agentic_workflow_durable_state_mcp_routing,
)


class TestEnterpriseAgenticWorkflowDurableStateMcpRoutingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_agentic_workflow_durable_state_mcp_routing()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_agentic_workflow_durable_state_mcp_routing_selection",
        )
        self.assertEqual(data["round"], 371)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_hierarchical_70b_planner_8b_executor_append_only_kv"]["task_completion_rate_pct"],
            tiers["tier1_monolithic_180_tools_unbounded_react_loop"]["task_completion_rate_pct"],
        )
        self.assertGreater(
            data["summary"]["tier4_cost_savings_vs_tier1_pct"],
            90.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_agentic_workflow_durable_state_mcp_routing_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_agentic_workflow_durable_state_mcp_routing()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

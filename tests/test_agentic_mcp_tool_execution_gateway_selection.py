"""Unit tests for Round 52 Enterprise Agentic MCP Tool-Use & CodeAct Gateway Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.agentic_mcp_tool_execution_gateway_selection.run import (
    evaluate_agentic_mcp_tool_execution_gateway_matrix,
)


class TestAgenticMcpToolExecutionGatewaySelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_agentic_mcp_tool_execution_gateway_matrix()
        self.assertEqual(data["experiment_id"], "agentic_mcp_tool_execution_gateway_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier1_full_500_schema_serial_react_raw_json"]["total_tokens_per_task"],
            tiers["tier4_programmatic_codeact_sandbox_loop_output_projection"]["total_tokens_per_task"] * 20.0,
        )
        self.assertLessEqual(
            tiers["tier4_programmatic_codeact_sandbox_loop_output_projection"]["llm_roundtrips_per_task"],
            2,
        )
        self.assertGreater(
            data["summary"]["tier4_cost_reduction_vs_tier1_pct"],
            95.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "agentic_mcp_tool_execution_gateway_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_agentic_mcp_tool_execution_gateway_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

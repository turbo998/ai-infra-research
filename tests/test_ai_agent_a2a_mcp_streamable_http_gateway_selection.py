"""Unit tests for Enterprise Multi-Agent Interoperability Protocol (MCP vs. A2A) & Streamable HTTP Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_agent_a2a_mcp_streamable_http_gateway_selection.run import (
    evaluate_ai_agent_a2a_mcp_streamable_http_gateway_matrix,
)


class TestAiAgentA2aMcpStreamableHttpGatewaySelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_agent_a2a_mcp_streamable_http_gateway_matrix()
        self.assertEqual(data["experiment_id"], "ai_agent_a2a_mcp_streamable_http_gateway_selection")
        self.assertEqual(data["round"], 176)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_dual_layer_a2a_horizontal_mcp_vertical_mesh"]["multi_domain_task_success_pct"],
            90.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_context_token_reduction_pct"],
            90.0,
        )
        self.assertGreater(
            data["summary"]["tier4_monthly_savings_vs_tier1_usd"],
            120000.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_agent_a2a_mcp_streamable_http_gateway_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_agent_a2a_mcp_streamable_http_gateway_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()

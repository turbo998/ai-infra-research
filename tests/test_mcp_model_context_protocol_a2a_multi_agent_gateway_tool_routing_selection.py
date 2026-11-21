import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMCPModelContextProtocolA2AGatewaySelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "mcp_model_context_protocol_a2a_multi_agent_gateway_tool_routing_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "mcp_model_context_protocol_a2a_multi_agent_gateway_tool_routing_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_MCP_A2A_Gateway_Tool_RAG_Plus_Code_Mode_Batching"]["End_To_End_Task_P95_Latency_sec"],
            tiers["Tier_1_Direct_All_400_Tools_Injected_Sequential_Turns"]["End_To_End_Task_P95_Latency_sec"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "mcp-model-context-protocol-a2a-multi-agent-gateway-tool-routing-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "mcp-model-context-protocol-a2a-multi-agent-gateway-tool-routing-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "mcp-model-context-protocol-a2a-multi-agent-gateway-tool-routing-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "mcp-model-context-protocol-a2a-multi-agent-gateway-tool-routing-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "mcp-model-context-protocol-a2a-multi-agent-gateway-tool-routing-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()

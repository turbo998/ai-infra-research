from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseMcpGatewayApimEnvoySemanticToolRouterOboAuthSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_mcp_gateway_apim_envoy_semantic_tool_router_obo_auth_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["tool_selection_accuracy_pct"], 96.0)
        self.assertGreater(data["tiers"][3]["prefix_kv_hit_rate_pct"], 90.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-mcp-gateway-apim-envoy-semantic-tool-router-obo-auth-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-mcp-gateway-apim-envoy-semantic-tool-router-obo-auth-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-mcp-gateway-apim-envoy-semantic-tool-router-obo-auth-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "mcp-gateway-apim-envoy-semantic-tool-router-obo-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-mcp-gateway-apim-envoy-semantic-tool-router-obo-auth-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-mcp-gateway-apim-envoy-semantic-tool-router-obo-auth-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseCustomerServiceCopilotStudioServicenowAgentforceCustomTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_customer_service_copilot_studio_servicenow_agentforce_custom_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["autonomous_deflection_rate_pct"], 85.0)
        self.assertLess(data["tiers"][3]["blended_cost_per_conversation_usd"], 0.02)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-customer-service-copilot-studio-servicenow-agentforce-custom-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-customer-service-copilot-studio-servicenow-agentforce-custom-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-customer-service-copilot-studio-servicenow-agentforce-custom-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "customer-service-copilot-studio-servicenow-agentforce-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-customer-service-copilot-studio-servicenow-agentforce-custom-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-customer-service-copilot-studio-servicenow-agentforce-custom-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

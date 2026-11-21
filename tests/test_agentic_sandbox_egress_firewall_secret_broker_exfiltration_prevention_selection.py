from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAgenticSandboxEgressFirewallSecretBrokerExfiltrationPreventionSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "agentic_sandbox_egress_firewall_secret_broker_exfiltration_prevention_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["data_exfiltration_block_rate_pct"], 99.0)
        self.assertEqual(data["tiers"][3]["raw_credential_theft_risk_pct"], 0.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "agentic-sandbox-egress-firewall-secret-broker-exfiltration-prevention-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "agentic-sandbox-egress-firewall-secret-broker-exfiltration-prevention-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "agentic-sandbox-egress-firewall-secret-broker-exfiltration-prevention-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "agentic-sandbox-egress-firewall-secret-broker-exfiltration-prevention-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "agentic-sandbox-egress-firewall-secret-broker-exfiltration-prevention-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

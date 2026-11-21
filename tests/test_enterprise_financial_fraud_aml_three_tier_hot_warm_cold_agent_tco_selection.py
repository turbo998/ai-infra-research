from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseFinancialFraudAmlThreeTierHotWarmColdAgentTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_financial_fraud_aml_three_tier_hot_warm_cold_agent_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLessEqual(data["tiers"][3]["payment_switch_p99_ms"], 20.0)
        self.assertLessEqual(data["tiers"][3]["false_positive_card_block_rate_pct"], 1.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-financial-fraud-aml-three-tier-hot-warm-cold-agent-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-financial-fraud-aml-three-tier-hot-warm-cold-agent-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-financial-fraud-aml-three-tier-hot-warm-cold-agent-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "financial-fraud-aml-three-tier-hot-warm-cold-agent-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-financial-fraud-aml-three-tier-hot-warm-cold-agent-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-financial-fraud-aml-three-tier-hot-warm-cold-agent-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

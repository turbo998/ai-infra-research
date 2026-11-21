from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiGuardrailsAzureContentSafetyBedrockLlamaguardPresidioTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_guardrails_azure_content_safety_bedrock_llamaguard_presidio_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLessEqual(data["tiers"][3]["added_ttft_latency_ms"], 15)
        self.assertGreater(data["tiers"][3]["injection_block_rate_pct"], 99.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-ai-guardrails-azure-content-safety-bedrock-llamaguard-presidio-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-ai-guardrails-azure-content-safety-bedrock-llamaguard-presidio-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-ai-guardrails-azure-content-safety-bedrock-llamaguard-presidio-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "ai-guardrails-content-safety-bedrock-llamaguard-presidio-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-ai-guardrails-azure-content-safety-bedrock-llamaguard-presidio-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-ai-guardrails-azure-content-safety-bedrock-llamaguard-presidio-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

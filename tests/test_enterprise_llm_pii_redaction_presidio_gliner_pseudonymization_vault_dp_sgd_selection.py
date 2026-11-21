from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLlmPiiRedactionPresidioGlinerPseudonymizationVaultDpSgdSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "enterprise_llm_pii_redaction_presidio_gliner_pseudonymization_vault_dp_sgd_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["pii_detection_recall_pct"], 98.0)
        self.assertGreater(data["tiers"][3]["downstream_llm_task_utility_pct"], 94.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-llm-pii-redaction-presidio-gliner-pseudonymization-vault-dp-sgd-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-llm-pii-redaction-presidio-gliner-pseudonymization-vault-dp-sgd-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-llm-pii-redaction-presidio-gliner-pseudonymization-vault-dp-sgd-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-llm-pii-redaction-presidio-gliner-pseudonymization-vault-dp-sgd-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-llm-pii-redaction-presidio-gliner-pseudonymization-vault-dp-sgd-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

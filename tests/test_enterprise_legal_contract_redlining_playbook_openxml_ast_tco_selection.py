from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLegalContractRedliningPlaybookOpenxmlAstTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_legal_contract_redlining_playbook_openxml_ast_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["cross_clause_poison_pill_recall_pct"], 98.0)
        self.assertGreater(data["tiers"][3]["counsel_one_click_acceptance_rate_pct"], 94.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-legal-contract-redlining-playbook-openxml-ast-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-legal-contract-redlining-playbook-openxml-ast-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-legal-contract-redlining-playbook-openxml-ast-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "legal-contract-redlining-playbook-openxml-ast-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-legal-contract-redlining-playbook-openxml-ast-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-legal-contract-redlining-playbook-openxml-ast-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseFinancialDeepResearchDueDiligenceAgentDagCitationTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_financial_deep_research_due_diligence_agent_dag_citation_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["numerical_audit_accuracy_pct"], 99.0)
        self.assertLessEqual(data["tiers"][3]["wall_clock_seconds"], 120)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-financial-deep-research-due-diligence-agent-dag-citation-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-financial-deep-research-due-diligence-agent-dag-citation-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-financial-deep-research-due-diligence-agent-dag-citation-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "financial-deep-research-due-diligence-agent-dag-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-financial-deep-research-due-diligence-agent-dag-citation-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-financial-deep-research-due-diligence-agent-dag-citation-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

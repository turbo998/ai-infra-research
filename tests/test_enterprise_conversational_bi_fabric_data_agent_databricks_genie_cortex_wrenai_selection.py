from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseConversationalBiFabricDataAgentDatabricksGenieCortexWrenaiSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_conversational_bi_fabric_data_agent_databricks_genie_cortex_wrenai_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["business_metric_accuracy_pct"], 94.0)
        self.assertEqual(data["tiers"][3]["rls_violation_risk_pct"], 0.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-conversational-bi-fabric-data-agent-databricks-genie-cortex-wrenai-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-conversational-bi-fabric-data-agent-databricks-genie-cortex-wrenai-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-conversational-bi-fabric-data-agent-databricks-genie-cortex-wrenai-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "conversational-bi-fabric-iq-databricks-genie-cortex-wrenai-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-conversational-bi-fabric-data-agent-databricks-genie-cortex-wrenai-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-conversational-bi-fabric-data-agent-databricks-genie-cortex-wrenai-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

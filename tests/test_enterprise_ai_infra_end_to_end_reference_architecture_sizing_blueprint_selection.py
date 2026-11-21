from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiInfraEndToEndReferenceArchitectureSizingBlueprintSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "enterprise_ai_infra_end_to_end_reference_architecture_sizing_blueprint_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["cluster_24h_effective_goodput_pct"], 90.0)
        self.assertLess(data["tiers"][3]["blended_cost_per_1m_tokens_usd"], 0.30)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-ai-infra-end-to-end-reference-architecture-sizing-blueprint-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-ai-infra-end-to-end-reference-architecture-sizing-blueprint-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-ai-infra-end-to-end-reference-architecture-sizing-blueprint-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-ai-infra-end-to-end-reference-architecture-sizing-blueprint-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-ai-infra-end-to-end-reference-architecture-sizing-blueprint-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

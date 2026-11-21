from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiCodingPlatformGithubCopilotVsHybridFimAgentRoiSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_coding_platform_github_copilot_vs_hybrid_fim_agent_roi_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLessEqual(data["tiers"][3]["fim_p95_latency_ms"], 120)
        self.assertGreater(data["tiers"][3]["fim_acceptance_rate_pct"], 35.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-ai-coding-platform-github-copilot-vs-hybrid-fim-agent-roi-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-ai-coding-platform-github-copilot-vs-hybrid-fim-agent-roi-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-ai-coding-platform-github-copilot-vs-hybrid-fim-agent-roi-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "ai-coding-platform-copilot-vs-hybrid-fim-agent-roi-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-ai-coding-platform-github-copilot-vs-hybrid-fim-agent-roi-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-ai-coding-platform-github-copilot-vs-hybrid-fim-agent-roi-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

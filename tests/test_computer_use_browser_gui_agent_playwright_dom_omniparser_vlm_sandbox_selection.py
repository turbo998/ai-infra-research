from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestComputerUseBrowserGuiAgentPlaywrightDomOmniparserVlmSandboxSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "computer_use_browser_gui_agent_playwright_dom_omniparser_vlm_sandbox_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["task_success_rate_pct"], 90.0)
        self.assertLess(data["tiers"][3]["avg_step_latency_ms"], 600.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "computer-use-browser-gui-agent-playwright-dom-omniparser-vlm-sandbox-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "computer-use-browser-gui-agent-playwright-dom-omniparser-vlm-sandbox-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "computer-use-browser-gui-agent-playwright-dom-omniparser-vlm-sandbox-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "computer-use-browser-gui-agent-playwright-dom-omniparser-vlm-sandbox-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "computer-use-browser-gui-agent-playwright-dom-omniparser-vlm-sandbox-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

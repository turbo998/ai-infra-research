from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestCloudBrowserFleetPlaywrightCdpPoolFirecrackerSnapshotIsolationSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "cloud_browser_fleet_playwright_cdp_pool_firecracker_snapshot_isolation_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["browser_ready_startup_ms"], 180.0)
        self.assertLess(data["tiers"][3]["session_crash_or_oom_rate_pct"], 0.1)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "cloud-browser-fleet-playwright-cdp-pool-firecracker-snapshot-isolation-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "cloud-browser-fleet-playwright-cdp-pool-firecracker-snapshot-isolation-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "cloud-browser-fleet-playwright-cdp-pool-firecracker-snapshot-isolation-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "cloud-browser-fleet-playwright-cdp-pool-firecracker-snapshot-isolation-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "cloud-browser-fleet-playwright-cdp-pool-firecracker-snapshot-isolation-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

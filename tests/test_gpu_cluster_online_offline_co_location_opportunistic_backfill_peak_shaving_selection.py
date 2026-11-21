from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGpuClusterOnlineOfflineCoLocationOpportunisticBackfillPeakShavingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "gpu_cluster_online_offline_co_location_opportunistic_backfill_peak_shaving_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["cluster_24h_avg_utilization_pct"], 90.0)
        self.assertLess(data["tiers"][3]["online_p99_ttft_ms"], 400.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "gpu-cluster-online-offline-co-location-opportunistic-backfill-peak-shaving-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "gpu-cluster-online-offline-co-location-opportunistic-backfill-peak-shaving-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "gpu-cluster-online-offline-co-location-opportunistic-backfill-peak-shaving-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "gpu-cluster-online-offline-co-location-opportunistic-backfill-peak-shaving-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "gpu-cluster-online-offline-co-location-opportunistic-backfill-peak-shaving-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

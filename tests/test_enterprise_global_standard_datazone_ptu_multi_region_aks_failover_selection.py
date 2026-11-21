from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseGlobalStandardDatazonePtuMultiRegionAksFailoverSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_global_standard_datazone_ptu_multi_region_aks_failover_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["burst_success_rate_pct"], 99.9)
        self.assertLess(data["tiers"][3]["regional_failover_rto_seconds"], 2.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-global-standard-datazone-ptu-multi-region-aks-failover-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-global-standard-datazone-ptu-multi-region-aks-failover-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-global-standard-datazone-ptu-multi-region-aks-failover-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "global-standard-datazone-ptu-aks-failover-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-global-standard-datazone-ptu-multi-region-aks-failover-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-global-standard-datazone-ptu-multi-region-aks-failover-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

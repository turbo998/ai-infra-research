from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseIndustrialManufacturingAoiPredictiveMaintenanceEdgeCloudSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_industrial_manufacturing_aoi_predictive_maintenance_edge_cloud_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLessEqual(data["tiers"][3]["inline_aoi_latency_ms"], 10.0)
        self.assertLessEqual(data["tiers"][3]["false_reject_overkill_rate_pct"], 2.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-industrial-manufacturing-aoi-predictive-maintenance-edge-cloud-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-industrial-manufacturing-aoi-predictive-maintenance-edge-cloud-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-industrial-manufacturing-aoi-predictive-maintenance-edge-cloud-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "industrial-manufacturing-aoi-predictive-maintenance-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-industrial-manufacturing-aoi-predictive-maintenance-edge-cloud-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-industrial-manufacturing-aoi-predictive-maintenance-edge-cloud-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

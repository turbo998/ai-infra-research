from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLlmCapacityPlanningTpmTpsThreeWatermarksPtuTcoSizingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_llm_capacity_planning_tpm_tps_three_watermarks_ptu_tco_sizing_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["peak_p99_sla_attainment_pct"], 99.0)
        self.assertLess(data["tiers"][3]["monthly_tco_usd"], 55000)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-llm-capacity-planning-tpm-tps-three-watermarks-ptu-tco-sizing-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-llm-capacity-planning-tpm-tps-three-watermarks-ptu-tco-sizing-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-llm-capacity-planning-tpm-tps-three-watermarks-ptu-tco-sizing-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "gpu-capacity-tpm-tps-ptu-tco-calculator.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-llm-capacity-planning-tpm-tps-three-watermarks-ptu-tco-sizing-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-llm-capacity-planning-tpm-tps-three-watermarks-ptu-tco-sizing-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

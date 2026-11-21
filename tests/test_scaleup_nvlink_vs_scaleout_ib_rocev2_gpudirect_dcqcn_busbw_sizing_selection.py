from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestScaleupNvlinkVsScaleoutIbRocev2GpudirectDcqcnBusbwSizingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "scaleup_nvlink_vs_scaleout_ib_rocev2_gpudirect_dcqcn_busbw_sizing_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["achieved_busbw_gbps"], 45.0)
        self.assertGreater(data["tiers"][3]["line_rate_efficiency_pct"], 95.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "scaleup-nvlink-vs-scaleout-ib-rocev2-gpudirect-dcqcn-busbw-sizing-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "scaleup-nvlink-vs-scaleout-ib-rocev2-gpudirect-dcqcn-busbw-sizing-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "scaleup-nvlink-vs-scaleout-ib-rocev2-gpudirect-dcqcn-busbw-sizing-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "nvlink-ib-rocev2-gpudirect-network-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "scaleup-nvlink-vs-scaleout-ib-rocev2-gpudirect-dcqcn-busbw-sizing-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "scaleup-nvlink-vs-scaleout-ib-rocev2-gpudirect-dcqcn-busbw-sizing-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

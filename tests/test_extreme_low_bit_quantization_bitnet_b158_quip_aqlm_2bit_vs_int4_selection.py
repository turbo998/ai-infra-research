from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestExtremeLowBitQuantizationBitnetB158QuipAqlm2bitVsInt4Selection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "extreme_low_bit_quantization_bitnet_b158_quip_aqlm_2bit_vs_int4_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["weight_footprint_70b_gb"], 16.5)
        self.assertGreater(data["tiers"][3]["energy_reduction_pct"], 70.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "extreme-low-bit-quantization-bitnet-b158-quip-aqlm-2bit-vs-int4-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "extreme-low-bit-quantization-bitnet-b158-quip-aqlm-2bit-vs-int4-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "extreme-low-bit-quantization-bitnet-b158-quip-aqlm-2bit-vs-int4-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "extreme-low-bit-quantization-bitnet-b158-quip-aqlm-2bit-vs-int4-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "extreme-low-bit-quantization-bitnet-b158-quip-aqlm-2bit-vs-int4-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

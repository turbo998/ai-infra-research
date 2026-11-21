from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAiContentProvenanceWatermarkingC2paSynthidTreeRingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "ai_content_provenance_watermarking_c2pa_synthid_tree_ring_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["survival_after_social_media_and_jpeg_pct"], 97.0)
        self.assertTrue(data["tiers"][3]["cryptographic_non_repudiation"])

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "ai-content-provenance-watermarking-c2pa-synthid-tree-ring-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "ai-content-provenance-watermarking-c2pa-synthid-tree-ring-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "ai-content-provenance-watermarking-c2pa-synthid-tree-ring-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "ai-content-provenance-watermarking-c2pa-synthid-tree-ring-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "ai-content-provenance-watermarking-c2pa-synthid-tree-ring-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMultimodalVisualTokenPruningFastvTomeDynamicTilingVlmSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "multimodal_visual_token_pruning_fastv_tome_dynamic_tiling_vlm_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["prefill_ttft_ms"], 500.0)
        self.assertGreater(data["tiers"][3]["docvqa_chart_accuracy_pct"], 93.5)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "multimodal-visual-token-pruning-fastv-tome-dynamic-tiling-vlm-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "multimodal-visual-token-pruning-fastv-tome-dynamic-tiling-vlm-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "multimodal-visual-token-pruning-fastv-tome-dynamic-tiling-vlm-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "multimodal-visual-token-pruning-fastv-tome-dynamic-tiling-vlm-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "multimodal-visual-token-pruning-fastv-tome-dynamic-tiling-vlm-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAdaptiveDepthLayerSkippingLayerskipMixtureOfDepthsKvCacheSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "adaptive_depth_layer_skipping_layerskip_mixture_of_depths_kv_cache_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["wall_clock_decode_speedup"], 1.8)
        self.assertEqual(data["tiers"][3]["benchmark_accuracy_pct"], 82.6)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "adaptive-depth-layer-skipping-layerskip-mixture-of-depths-kv-cache-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "adaptive-depth-layer-skipping-layerskip-mixture-of-depths-kv-cache-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "adaptive-depth-layer-skipping-layerskip-mixture-of-depths-kv-cache-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "adaptive-depth-layer-skipping-layerskip-mixture-of-depths-kv-cache-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "adaptive-depth-layer-skipping-layerskip-mixture-of-depths-kv-cache-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

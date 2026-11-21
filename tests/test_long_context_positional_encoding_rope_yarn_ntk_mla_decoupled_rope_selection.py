from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLongContextPositionalEncodingRopeYarnNtkMlaDecoupledRopeSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "long_context_positional_encoding_rope_yarn_ntk_mla_decoupled_rope_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["niah_128k_recall_pct"], 98.5)
        self.assertEqual(data["tiers"][3]["kv_cache_elements_per_token_layer"], 576)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "long-context-positional-encoding-rope-yarn-ntk-mla-decoupled-rope-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "long-context-positional-encoding-rope-yarn-ntk-mla-decoupled-rope-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "long-context-positional-encoding-rope-yarn-ntk-mla-decoupled-rope-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "long-context-positional-encoding-rope-yarn-ntk-mla-decoupled-rope-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "long-context-positional-encoding-rope-yarn-ntk-mla-decoupled-rope-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

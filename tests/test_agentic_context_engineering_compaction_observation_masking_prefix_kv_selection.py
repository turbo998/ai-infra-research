from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAgenticContextEngineeringCompactionObservationMaskingPrefixKvSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "agentic_context_engineering_compaction_observation_masking_prefix_kv_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["prefix_kv_cache_hit_rate_pct"], 90.0)
        self.assertGreater(data["tiers"][3]["long_horizon_50_turn_success_pct"], 92.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "agentic-context-engineering-compaction-observation-masking-prefix-kv-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "agentic-context-engineering-compaction-observation-masking-prefix-kv-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "agentic-context-engineering-compaction-observation-masking-prefix-kv-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "agentic-context-engineering-compaction-observation-masking-prefix-kv-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "agentic-context-engineering-compaction-observation-masking-prefix-kv-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestPostTrainingAlignmentDpoSimpoOrpoKtoGrpoMemoryTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "post_training_alignment_dpo_simpo_orpo_kto_grpo_memory_tco_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][2]["peak_hbm_gb_per_gpu"], 60.0)
        self.assertGreater(data["tiers"][3]["domain_reasoning_and_alignment_score_pct"], 92.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "post-training-alignment-dpo-simpo-orpo-kto-grpo-memory-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "post-training-alignment-dpo-simpo-orpo-kto-grpo-memory-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "post-training-alignment-dpo-simpo-orpo-kto-grpo-memory-tco-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "post-training-alignment-dpo-simpo-orpo-kto-grpo-memory-tco-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "post-training-alignment-dpo-simpo-orpo-kto-grpo-memory-tco-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

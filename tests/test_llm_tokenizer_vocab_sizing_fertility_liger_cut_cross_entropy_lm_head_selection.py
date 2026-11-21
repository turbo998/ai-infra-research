from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmTokenizerVocabSizingFertilityLigerCutCrossEntropyLmHeadSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "llm_tokenizer_vocab_sizing_fertility_liger_cut_cross_entropy_lm_head_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["lm_head_logits_peak_hbm_gb"], 0.5)
        self.assertGreater(data["tiers"][3]["relative_inference_speedup_zh_code"], 2.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "llm-tokenizer-vocab-sizing-fertility-liger-cut-cross-entropy-lm-head-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "llm-tokenizer-vocab-sizing-fertility-liger-cut-cross-entropy-lm-head-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "llm-tokenizer-vocab-sizing-fertility-liger-cut-cross-entropy-lm-head-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "llm-tokenizer-vocab-sizing-fertility-liger-cut-cross-entropy-lm-head-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "llm-tokenizer-vocab-sizing-fertility-liger-cut-cross-entropy-lm-head-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

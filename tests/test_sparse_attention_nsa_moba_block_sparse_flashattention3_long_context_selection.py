from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSparseAttentionNsaMobaBlockSparseFlashattention3LongContextSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "sparse_attention_nsa_moba_block_sparse_flashattention3_long_context_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["prefill_speedup_vs_dense"], 8.5)
        self.assertGreater(data["tiers"][3]["ruler_128k_accuracy_pct"], 99.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

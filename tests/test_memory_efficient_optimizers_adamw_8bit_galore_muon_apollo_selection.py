from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMemoryEfficientOptimizersAdamw8bitGaloreMuonApolloSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "memory_efficient_optimizers_adamw_8bit_galore_muon_apollo_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["token_efficiency_speedup_to_target_loss"], 1.35)
        self.assertLess(data["tiers"][2]["total_70b_optimizer_state_gb"], 200.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

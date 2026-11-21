from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestActivationCheckpointingSelectiveRecomputationPcieOffloadMfuSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "activation_checkpointing_selective_recomputation_pcie_offload_mfu_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["activation_hbm_gb_per_gpu"], 10.0)
        self.assertGreater(data["tiers"][3]["effective_model_mfu_pct"], 57.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "activation-checkpointing-selective-recomputation-pcie-offload-mfu-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "activation-checkpointing-selective-recomputation-pcie-offload-mfu-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "activation-checkpointing-selective-recomputation-pcie-offload-mfu-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "activation-checkpointing-selective-recomputation-pcie-offload-mfu-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "activation-checkpointing-selective-recomputation-pcie-offload-mfu-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

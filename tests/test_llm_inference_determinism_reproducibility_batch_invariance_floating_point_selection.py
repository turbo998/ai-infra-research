from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmInferenceDeterminismReproducibilityBatchInvarianceFloatingPointSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "llm_inference_determinism_reproducibility_batch_invariance_floating_point_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertEqual(data["tiers"][3]["bitwise_reproducibility_under_load_pct"], 100.0)
        self.assertLess(data["tiers"][3]["throughput_overhead_pct"], 10.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "llm-inference-determinism-reproducibility-batch-invariance-floating-point-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "llm-inference-determinism-reproducibility-batch-invariance-floating-point-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "llm-inference-determinism-reproducibility-batch-invariance-floating-point-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "llm-inference-determinism-reproducibility-batch-invariance-floating-point-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "llm-inference-determinism-reproducibility-batch-invariance-floating-point-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

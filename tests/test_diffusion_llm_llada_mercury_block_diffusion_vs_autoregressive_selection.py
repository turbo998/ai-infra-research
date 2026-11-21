from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestDiffusionLlmLladaMercuryBlockDiffusionVsAutoregressiveSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "diffusion_llm_llada_mercury_block_diffusion_vs_autoregressive_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["per_user_output_tps"], 900.0)
        self.assertTrue(data["tiers"][3]["supports_causal_prefix_kv_cache"])

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "diffusion-llm-llada-mercury-block-diffusion-vs-autoregressive-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "diffusion-llm-llada-mercury-block-diffusion-vs-autoregressive-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "diffusion-llm-llada-mercury-block-diffusion-vs-autoregressive-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "diffusion-llm-llada-mercury-block-diffusion-vs-autoregressive-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "diffusion-llm-llada-mercury-block-diffusion-vs-autoregressive-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

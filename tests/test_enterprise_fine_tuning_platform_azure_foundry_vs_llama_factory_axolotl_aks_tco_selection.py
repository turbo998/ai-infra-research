from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseFineTuningPlatformAzureFoundryVsLlamaFactoryAxolotlAksTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_fine_tuning_platform_azure_foundry_vs_llama_factory_axolotl_aks_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertEqual(data["tiers"][3]["serving_gpus_required"], 8)
        self.assertLess(data["tiers"][3]["monthly_training_plus_serving_tco_usd"], 25000)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-fine-tuning-platform-azure-foundry-vs-llama-factory-axolotl-aks-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-fine-tuning-platform-azure-foundry-vs-llama-factory-axolotl-aks-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-fine-tuning-platform-azure-foundry-vs-llama-factory-axolotl-aks-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "fine-tuning-foundry-vs-llama-factory-axolotl-aks-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-fine-tuning-platform-azure-foundry-vs-llama-factory-axolotl-aks-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-fine-tuning-platform-azure-foundry-vs-llama-factory-axolotl-aks-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

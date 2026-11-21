from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmObservabilityOpentelemetryGenaiLangfusePhoenixFinopsChargebackSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "llm_observability_opentelemetry_genai_langfuse_phoenix_finops_chargeback_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["added_hot_path_p99_latency_ms"], 0.5)
        self.assertLess(data["tiers"][3]["finops_chargeback_billing_error_pct"], 0.1)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "llm-observability-opentelemetry-genai-langfuse-phoenix-finops-chargeback-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "llm-observability-opentelemetry-genai-langfuse-phoenix-finops-chargeback-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "llm-observability-opentelemetry-genai-langfuse-phoenix-finops-chargeback-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "llm-observability-opentelemetry-genai-langfuse-phoenix-finops-chargeback-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "llm-observability-opentelemetry-genai-langfuse-phoenix-finops-chargeback-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

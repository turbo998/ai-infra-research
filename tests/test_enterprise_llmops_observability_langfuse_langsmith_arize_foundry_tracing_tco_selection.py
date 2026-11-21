from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLlmopsObservabilityLangfuseLangsmithArizeFoundryTracingTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_llmops_observability_langfuse_langsmith_arize_foundry_tracing_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertEqual(data["tiers"][3]["bad_case_capture_rate_pct"], 100.0)
        self.assertLess(data["tiers"][3]["monthly_observability_tco_usd"], 6000)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-llmops-observability-langfuse-langsmith-arize-foundry-tracing-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-llmops-observability-langfuse-langsmith-arize-foundry-tracing-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-llmops-observability-langfuse-langsmith-arize-foundry-tracing-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "llmops-observability-langfuse-langsmith-arize-foundry-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-llmops-observability-langfuse-langsmith-arize-foundry-tracing-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-llmops-observability-langfuse-langsmith-arize-foundry-tracing-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

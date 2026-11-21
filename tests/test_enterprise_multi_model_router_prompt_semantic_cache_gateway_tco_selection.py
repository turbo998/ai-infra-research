from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseMultiModelRouterPromptSemanticCacheGatewayTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_multi_model_router_prompt_semantic_cache_gateway_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["cost_per_1m_requests_usd"], 1000)
        self.assertGreater(data["tiers"][3]["task_accuracy_pct"], 95.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-multi-model-router-prompt-semantic-cache-gateway-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-multi-model-router-prompt-semantic-cache-gateway-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-multi-model-router-prompt-semantic-cache-gateway-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "multi-model-router-prompt-semantic-cache-gateway-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-multi-model-router-prompt-semantic-cache-gateway-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-multi-model-router-prompt-semantic-cache-gateway-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

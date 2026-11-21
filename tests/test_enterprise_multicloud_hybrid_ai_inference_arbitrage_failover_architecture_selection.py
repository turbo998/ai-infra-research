from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseMulticloudHybridAiInferenceArbitrageFailoverArchitectureSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_multicloud_hybrid_ai_inference_arbitrage_failover_architecture_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["cross_cloud_prefix_hit_rate_pct"], 90.0)
        self.assertLess(data["tiers"][3]["blended_cost_per_1m_tokens_usd"], 0.25)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-multicloud-hybrid-ai-inference-arbitrage-failover-architecture-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-multicloud-hybrid-ai-inference-arbitrage-failover-architecture-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-multicloud-hybrid-ai-inference-arbitrage-failover-architecture-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "multicloud-hybrid-ai-inference-arbitrage-failover-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-multicloud-hybrid-ai-inference-arbitrage-failover-architecture-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-multicloud-hybrid-ai-inference-arbitrage-failover-architecture-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

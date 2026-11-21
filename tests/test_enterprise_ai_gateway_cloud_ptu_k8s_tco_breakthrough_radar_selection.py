from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiGatewayCloudPtuK8sTcoBreakthroughRadarSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_gateway_cloud_ptu_k8s_tco_breakthrough_radar_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["prefix_kv_cache_hit_rate_pct"], 90.0)
        self.assertLess(data["tiers"][3]["blended_cost_per_1m_tokens_usd"], 0.30)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-ai-gateway-cloud-ptu-k8s-tco-breakthrough-radar-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-ai-gateway-cloud-ptu-k8s-tco-breakthrough-radar-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-ai-gateway-cloud-ptu-k8s-tco-breakthrough-radar-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "ai-gateway-cloud-ptu-k8s-tco-radar-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-ai-gateway-cloud-ptu-k8s-tco-breakthrough-radar-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-ai-gateway-cloud-ptu-k8s-tco-breakthrough-radar-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

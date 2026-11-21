from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseRetailMultimodalSearchShoppingAgentRedisMilvusTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_retail_multimodal_search_shopping_agent_redis_milvus_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLessEqual(data["tiers"][3]["search_p99_latency_ms"], 70)
        self.assertEqual(data["tiers"][3]["stale_price_or_out_of_stock_rate_pct"], 0.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "retail-multimodal-search-shopping-agent-milvus-redis-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

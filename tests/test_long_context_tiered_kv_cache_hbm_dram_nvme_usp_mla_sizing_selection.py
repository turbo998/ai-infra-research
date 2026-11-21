from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLongContextTieredKvCacheHbmDramNvmeUspMlaSizingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "long_context_tiered_kv_cache_hbm_dram_nvme_usp_mla_sizing_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["multiturn_restore_ttft_ms"], 100.0)
        self.assertGreater(data["tiers"][3]["max_concurrent_128k_sessions"], 500)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "long-context-tiered-kv-usp-mla-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

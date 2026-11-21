from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestPdDisaggregationElasticRatioHeterogeneousGpuKvBandwidthSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "pd_disaggregation_elastic_ratio_heterogeneous_gpu_kv_bandwidth_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["diurnal_blended_goodput_tok_s"], 5500)
        self.assertLess(data["tiers"][3]["cost_per_1m_goodput_tokens_usd"], 0.30)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "pd-disaggregation-elastic-ratio-heterogeneous-gpu-kv-bandwidth-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "pd-disaggregation-elastic-ratio-heterogeneous-gpu-kv-bandwidth-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "pd-disaggregation-elastic-ratio-heterogeneous-gpu-kv-bandwidth-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "pd-disaggregation-ratio-kv-bandwidth-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "pd-disaggregation-elastic-ratio-heterogeneous-gpu-kv-bandwidth-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "pd-disaggregation-elastic-ratio-heterogeneous-gpu-kv-bandwidth-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseEdgeCloudVideoVlmJetsonL40sCascadeBandwidthTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_edge_cloud_video_vlm_jetson_l40s_cascade_bandwidth_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["wan_uplink_mbps"], 60)
        self.assertGreater(data["tiers"][3]["complex_event_f1_score_pct"], 95.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-edge-cloud-video-vlm-jetson-l40s-cascade-bandwidth-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-edge-cloud-video-vlm-jetson-l40s-cascade-bandwidth-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-edge-cloud-video-vlm-jetson-l40s-cascade-bandwidth-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "edge-cloud-video-vlm-jetson-l40s-cascade-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-edge-cloud-video-vlm-jetson-l40s-cascade-bandwidth-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-edge-cloud-video-vlm-jetson-l40s-cascade-bandwidth-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

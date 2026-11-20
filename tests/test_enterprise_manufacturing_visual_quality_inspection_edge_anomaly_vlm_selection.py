import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseManufacturingVisualQualitySelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_manufacturing_visual_quality_inspection_edge_anomaly_vlm_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_script_check_and_invariants(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"], "synthetic + anonymized_reported")
        self.assertFalse(data["independently_reproduced"])
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_full_frame_cloud_vlm_streaming"]
        t4 = tiers["tier4_edge_patchcore_yolo_plc_plus_gray_zone_7b_vlm"]
        self.assertLess(t4["inline_plc_trigger_latency_ms"], 15.0)
        self.assertLess(t4["defect_escape_rate_pct"], 0.1)
        self.assertLess(t4["overkill_false_reject_rate_pct"], 1.0)
        self.assertLess(
            t4["monthly_inspection_scrap_and_compute_tco_usd"],
            t1["monthly_inspection_scrap_and_compute_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-manufacturing-visual-quality-inspection-edge-anomaly-vlm"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseFieldServiceTechnicianSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_field_service_technician_offline_edge_rag_schematic_dispatch_selection"
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
        t1 = tiers["tier1_cloud_only_api_rag_no_van_stock_check"]
        t4 = tiers["tier4_predispatch_van_stock_offline_slm_sqlite_vec_bom_pinning"]
        self.assertGreater(t4["first_time_fix_rate_ftfr_pct"], 95.0)
        self.assertEqual(t4["offline_dead_zone_availability_pct"], 100.0)
        self.assertLess(t4["wrong_spare_part_order_rate_pct"], 2.0)
        self.assertLess(
            t4["monthly_repeat_truck_roll_and_parts_tco_usd"],
            t1["monthly_repeat_truck_roll_and_parts_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-field-service-technician-offline-edge-rag-schematic-dispatch"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMultiRegionActiveActiveLlmHaDrSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "multi_region_active_active_llm_ha_dr_geo_prefix_failover_selection"
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
        t2 = tiers["tier2_active_passive_100pct_idle_hot_standby_gpu_cluster"]
        t4 = tiers["tier4_active_active_geo_sticky_plus_top_prefix_warm_sync_and_3tier_escape_hatch"]
        self.assertLess(t4["regional_outage_rto_sec"], 2.5)
        self.assertEqual(t4["failover_cascading_overload_drop_rate_pct"], 0.0)
        self.assertLess(t4["monthly_global_ha_dr_tco_usd"], t2["monthly_global_ha_dr_tco_usd"] * 0.45)

    def test_artifacts_exist(self) -> None:
        slug = "multi-region-active-active-llm-ha-dr-geo-prefix-failover"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

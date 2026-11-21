import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLlmTpmTpsLittlesLawCapacityPlanningSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_llm_tpm_tps_littles_law_capacity_planning_ptu_sizing_selection"
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
        t1 = tiers["tier1_daily_average_rps_sizing_ignoring_peak_and_output_weight"]
        t4 = tiers["tier4_littles_law_weighted_tpm_p80_base_plus_p20_paygo_spillover"]
        self.assertEqual(t4["peak_hour_http_429_and_sla_drop_rate_pct"], 0.0)
        self.assertGreater(t4["effective_cluster_utilization_pct"], 85.0)
        self.assertLess(
            t4["monthly_50rps_capacity_tco_usd"],
            t1["monthly_50rps_capacity_tco_usd"] * 0.40,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

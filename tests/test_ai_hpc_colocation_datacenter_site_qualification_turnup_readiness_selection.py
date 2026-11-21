import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAiHpcColocationSiteQualificationTurnupSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection"
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
        t1 = tiers["tier1_legacy_air_cooled_colo_ping_only_handover"]
        t4 = tiers["tier4_8stage_dlc_cdu_aligned_turnup_48h_burnin_90pct_busbw_gate"]
        self.assertLess(t4["turnup_to_stable_production_days"], 10.0)
        self.assertLess(t4["first_month_infant_mortality_job_abort_pct"], 1.0)
        self.assertGreater(t4["sustained_collective_busbw_pct_of_spec"], 94.0)
        self.assertLess(
            t4["monthly_downtime_rework_and_cooling_tco_usd"],
            t1["monthly_downtime_rework_and_cooling_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "ai-hpc-colocation-datacenter-site-qualification-turnup-readiness"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

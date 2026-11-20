import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiIncidentResponseSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_incident_response_kill_switch_degraded_fallback_quarantine_selection"
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
        t1 = tiers["tier1_git_redeploy_or_global_plug_pull"]
        t4 = tiers["tier4_granular_killswitch_chunk_quarantine_graceful_ladder"]
        self.assertLessEqual(t4["mean_time_to_mitigate_sec"], 2.0)
        self.assertGreater(t4["service_availability_during_incident_pct"], 94.0)
        self.assertEqual(t4["residual_poisoned_cache_hit_rate_pct"], 0.0)
        self.assertLess(
            t4["monthly_incident_blast_radius_tco_usd"],
            t1["monthly_incident_blast_radius_tco_usd"] * 0.12,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-ai-incident-response-kill-switch-degraded-fallback-quarantine"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

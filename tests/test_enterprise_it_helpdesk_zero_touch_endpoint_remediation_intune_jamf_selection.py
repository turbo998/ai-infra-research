import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseItHelpdeskEndpointRemediationSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_it_helpdesk_zero_touch_endpoint_remediation_intune_jamf_selection"
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
        t1 = tiers["tier1_readonly_kb_article_link_bot"]
        t4 = tiers["tier4_telemetry_prefetch_signed_runbook_mdm_agent"]
        self.assertGreater(t4["zero_touch_ticket_resolution_pct"], 78.0)
        self.assertEqual(t4["destructive_endpoint_command_risk_pct"], 0.0)
        self.assertLessEqual(t4["mean_time_to_resolution_minutes"], 1.5)
        self.assertLess(
            t4["monthly_it_helpdesk_tco_usd"],
            t1["monthly_it_helpdesk_tco_usd"] * 0.12,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-it-helpdesk-zero-touch-endpoint-remediation-intune-jamf"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

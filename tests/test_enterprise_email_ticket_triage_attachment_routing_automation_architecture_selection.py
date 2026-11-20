import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseEmailTicketTriageSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_email_ticket_triage_attachment_routing_automation_architecture_selection"
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
        t1 = tiers["tier1_raw_thread_frontier_llm_naive_autoreply"]
        t4 = tiers["tier4_rfc3834_loop_guard_quote_strip_tiered_triage"]
        self.assertEqual(t4["auto_reply_loop_incident_rate_pct"], 0.0)
        self.assertGreater(t4["triage_routing_accuracy_pct"], 96.0)
        self.assertGreaterEqual(t4["prompt_token_reduction_pct"], 70.0)
        self.assertLess(t4["monthly_total_tco_usd"], t1["monthly_total_tco_usd"] * 0.15)

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-email-ticket-triage-attachment-routing-automation-architecture"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

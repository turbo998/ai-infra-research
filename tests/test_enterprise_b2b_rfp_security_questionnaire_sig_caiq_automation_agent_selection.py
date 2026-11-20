import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseB2bRfpQuestionnaireSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_b2b_rfp_security_questionnaire_sig_caiq_automation_agent_selection"
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
        t1 = tiers["tier1_uncurated_wiki_and_past_proposals_rag"]
        t4 = tiers["tier4_canonical_qa_bank_ga_firewall_3band_router_excel_writeback"]
        self.assertEqual(t4["binding_over_commitment_error_rate_pct"], 0.0)
        self.assertGreater(t4["verified_auto_completion_rate_pct"], 90.0)
        self.assertLess(t4["rfp_turnaround_hours"], 1.0)
        self.assertLess(
            t4["monthly_presales_and_liability_tco_usd"],
            t1["monthly_presales_and_liability_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-b2b-rfp-security-questionnaire-sig-caiq-automation-agent"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

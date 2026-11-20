import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseWealthManagementAdvisorSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_wealth_management_advisor_copilot_suitability_ips_guardrail_selection"
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
        t1 = tiers["tier1_unconstrained_llm_portfolio_advice_and_email_drafting"]
        t4 = tiers["tier4_ips_apl_convex_solver_cio_house_view_finra2210_linter"]
        self.assertEqual(t4["reg_bi_mifid_suitability_violation_rate_pct"], 0.0)
        self.assertEqual(t4["finra_2210_promissory_language_violation_pct"], 0.0)
        self.assertLess(t4["advisor_meeting_prep_time_minutes"], 4.0)
        self.assertLess(
            t4["monthly_compliance_remediation_and_prep_tco_usd"],
            t1["monthly_compliance_remediation_and_prep_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-wealth-management-advisor-copilot-suitability-ips-guardrail"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

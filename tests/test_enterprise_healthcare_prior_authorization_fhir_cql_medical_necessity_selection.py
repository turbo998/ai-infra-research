import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseHealthcarePriorAuthorizationSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_healthcare_prior_authorization_fhir_cql_medical_necessity_selection"
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
        t1 = tiers["tier1_autonomous_blackbox_llm_approve_and_deny"]
        t4 = tiers["tier4_davinci_fhir_cql_engine_approve_only_md_gate"]
        self.assertGreater(t4["instant_auto_approval_rate_pct"], 75.0)
        self.assertEqual(t4["illegal_algorithmic_auto_denial_rate_pct"], 0.0)
        self.assertGreater(t4["clinical_criterion_evaluation_accuracy_pct"], 99.0)
        self.assertLess(
            t4["monthly_um_staffing_appeal_and_legal_tco_usd"],
            t1["monthly_um_staffing_appeal_and_legal_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-healthcare-prior-authorization-fhir-cql-medical-necessity"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseB2bOnboardingKybSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation_selection"
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
        t1 = tiers["tier1_manual_backoffice_review_email_chase"]
        t4 = tiers["tier4_preflight_gate_cross_doc_matrix_ubo_dag_solver"]
        self.assertGreater(t4["cross_doc_mismatch_detection_pct"], 99.5)
        self.assertEqual(t4["multilayer_ubo_math_error_rate_pct"], 0.0)
        self.assertLess(t4["onboarding_turnaround_hours"], 0.5)
        self.assertLess(
            t4["monthly_kyb_ops_and_dropoff_tco_usd"],
            t1["monthly_kyb_ops_and_dropoff_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

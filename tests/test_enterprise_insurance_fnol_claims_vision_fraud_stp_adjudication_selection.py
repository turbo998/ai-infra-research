import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseInsuranceFnolClaimsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection"
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
        t1 = tiers["tier1_direct_vlm_dollar_guess_no_fraud_gate"]
        t4 = tiers["tier4_c2pa_dire_fraud_gate_sam2_vin_catalog_opa_stp"]
        self.assertLess(t4["repair_cost_estimation_mape_pct"], 5.0)
        self.assertGreater(t4["synthetic_and_recycled_photo_fraud_catch_pct"], 99.0)
        self.assertGreater(t4["straight_through_processing_stp_rate_pct"], 65.0)
        self.assertLess(
            t4["monthly_fraud_leakage_and_adjuster_tco_usd"],
            t1["monthly_fraud_leakage_and_adjuster_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterprisePharmacovigilanceSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_pharmacovigilance_adverse_event_icsr_meddra_e2b_coding_selection"
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
        t1 = tiers["tier1_unconstrained_llm_meddra_code_generation"]
        t4 = tiers["tier4_day0_ime_gate_sapbert_meddra_pinning_e2b_schematron"]
        self.assertGreater(t4["meddra_pt_top1_coding_accuracy_pct"], 96.0)
        self.assertEqual(t4["invalid_or_retired_meddra_code_hallucination_pct"], 0.0)
        self.assertEqual(t4["late_15_day_expedited_submission_rate_pct"], 0.0)
        self.assertLess(
            t4["monthly_pv_cro_and_compliance_tco_usd"],
            t1["monthly_pv_cro_and_compliance_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-pharmacovigilance-adverse-event-icsr-meddra-e2b-coding"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

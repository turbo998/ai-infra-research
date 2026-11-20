import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseHrTalentMatchingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_hr_talent_matching_blind_skill_graph_adverse_impact_audit_selection"
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
        t1 = tiers["tier1_raw_resume_llm_scoring_unblinded"]
        t4 = tiers["tier4_blind_scrubber_esco_skill_evidence_graph_four_fifths_gate"]
        self.assertGreaterEqual(t4["eeoc_adverse_impact_ratio"], 0.90)
        self.assertGreater(t4["qualified_candidate_recall_pct"], 95.0)
        self.assertLess(t4["hiring_manager_false_positive_reject_pct"], 8.0)
        self.assertLess(
            t4["monthly_recruiting_and_compliance_tco_usd"],
            t1["monthly_recruiting_and_compliance_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-hr-talent-matching-blind-skill-graph-adverse-impact-audit"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

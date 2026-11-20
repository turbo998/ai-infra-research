import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLegalClmContractRedliningSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_legal_clm_contract_redlining_playbook_fallback_clause_agent_selection"
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
        t1 = tiers["tier1_plain_text_chat_freeform_llm_redwrite"]
        t4 = tiers["tier4_definition_ast_3tier_fallback_ooxml_track_changes"]
        self.assertGreater(t4["backdoor_definition_trap_catch_pct"], 99.0)
        self.assertEqual(t4["unapproved_liability_wording_rate_pct"], 0.0)
        self.assertLessEqual(t4["contract_turnaround_hours"], 0.25)
        self.assertLess(
            t4["monthly_legal_review_and_delay_tco_usd"],
            t1["monthly_legal_review_and_delay_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-legal-clm-contract-redlining-playbook-fallback-clause-agent"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiPrCodeReviewAppsecSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection"
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
        t1 = tiers["tier1_raw_git_diff_only_chatty_llm"]
        t4 = tiers["tier4_lsp_callgraph_sast_reachability_compile_verified_autofix"]
        self.assertGreater(t4["cross_file_bug_and_vuln_catch_pct"], 94.0)
        self.assertLess(t4["false_positive_and_nit_noise_rate_pct"], 5.0)
        self.assertGreater(t4["developer_autofix_accept_rate_pct"], 88.0)
        self.assertLess(
            t4["monthly_review_and_appsec_tco_usd"],
            t1["monthly_review_and_appsec_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-ai-pr-code-review-appsec-sast-reachability-autofix"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

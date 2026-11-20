import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseCloudFinopsRightsizingAgentSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_cloud_finops_k8s_gpu_rightsizing_commitment_portfolio_agent_selection"
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
        t1 = tiers["tier1_monthly_csv_spreadsheet_review"]
        t2 = tiers["tier2_naive_avg_cpu_gpu_auto_rightsizer"]
        t4 = tiers["tier4_focus_p999_vram_gitops_pr_rolling_ladder_portfolio"]
        self.assertEqual(t4["post_rightsizing_oom_outage_rate_pct"], 0.0)
        self.assertGreater(t2["post_rightsizing_oom_outage_rate_pct"], 20.0)
        self.assertGreater(t4["realized_net_cloud_savings_pct"], 30.0)
        self.assertLess(t4["effective_monthly_cloud_spend_usd"], t1["effective_monthly_cloud_spend_usd"] * 0.75)

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-cloud-finops-k8s-gpu-rightsizing-commitment-portfolio-agent"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

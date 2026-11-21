import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestCloudDeploymentPaygoVsPtuVsSelfHostedK8sSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "cloud_deployment_paygo_vs_provisioned_ptu_gsu_vs_self_hosted_k8s_tco_selection"
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
        t1 = tiers["tier1_100pct_serverless_paygo_at_15b_tokens_per_month"]
        t4 = tiers["tier4_hybrid_self_hosted_k8s_or_p80_ptu_base_plus_paygo_spillover_and_70_30_model_routing"]
        self.assertLess(t4["monthly_15b_tokens_total_tco_usd"], t1["monthly_15b_tokens_total_tco_usd"] * 0.30)
        self.assertGreaterEqual(t4["sla_availability_pct"], 99.9)

    def test_artifacts_exist(self) -> None:
        slug = "cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

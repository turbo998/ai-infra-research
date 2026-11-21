import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSovereignAIAirGappedOnPremApplianceSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "sovereign_ai_air_gapped_private_cloud_on_prem_appliance_deployment_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "sovereign_ai_air_gapped_private_cloud_on_prem_appliance_deployment_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_AirGap_Declarative_Cluster_Bundle_Harbor_HFMirror_KMS"]["On_Site_Turnup_Time_Hours"],
            tiers["Tier_1_Unmodified_Cloud_Native_Manifests_In_AirGap"]["On_Site_Turnup_Time_Hours"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "sovereign-ai-air-gapped-private-cloud-on-prem-appliance-deployment-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "sovereign-ai-air-gapped-private-cloud-on-prem-appliance-deployment-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "sovereign-ai-air-gapped-private-cloud-on-prem-appliance-deployment-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "sovereign-ai-air-gapped-private-cloud-on-prem-appliance-deployment-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "sovereign-ai-air-gapped-private-cloud-on-prem-appliance-deployment-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()

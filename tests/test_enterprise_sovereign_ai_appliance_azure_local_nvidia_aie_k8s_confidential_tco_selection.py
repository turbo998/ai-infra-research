from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseSovereignAiApplianceAzureLocalNvidiaAieK8sConfidentialTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_sovereign_ai_appliance_azure_local_nvidia_aie_k8s_confidential_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLessEqual(data["tiers"][3]["air_gapped_deployment_hours"], 5.0)
        self.assertLess(data["tiers"][3]["confidential_tee_throughput_tax_pct"], 5.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-sovereign-ai-appliance-azure-local-nvidia-aie-k8s-confidential-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-sovereign-ai-appliance-azure-local-nvidia-aie-k8s-confidential-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-sovereign-ai-appliance-azure-local-nvidia-aie-k8s-confidential-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "sovereign-ai-azure-local-nvidia-aie-k8s-confidential-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-sovereign-ai-appliance-azure-local-nvidia-aie-k8s-confidential-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-sovereign-ai-appliance-azure-local-nvidia-aie-k8s-confidential-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()

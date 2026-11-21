import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestConfidentialAiGpuTeeH200BlackwellPpcieSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "confidential_ai_gpu_tee_h200_blackwell_ppcie_attestation_selection"
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
        t1 = tiers["tier1_cpu_only_tee_or_naive_gpu_cc_with_frequent_host_pcie_transfers"]
        t4 = tiers["tier4_blackwell_b200_inline_tee_tdisp_line_rate_nvlink5_encryption"]
        self.assertGreater(t4["tee_throughput_retention_vs_baremetal_pct"], 98.0)
        self.assertLess(
            t4["cost_per_1m_confidential_tokens_usd"],
            t1["cost_per_1m_confidential_tokens_usd"] * 0.30,
        )

    def test_artifacts_exist(self) -> None:
        slug = "confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

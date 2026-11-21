import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMultiVendorAiAcceleratorsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "multi_vendor_ai_accelerators_nvidia_tpu_amd_mi325x_trainium2_selection"
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
        t1 = tiers["tier1_single_vendor_100pct_on_demand_gpu_lock_in_with_cuda_custom_ops"]
        t4 = tiers["tier4_multi_backend_vllm_sglang_gateway_portfolio_nvidia_tpu_mi325x_trn2"]
        self.assertGreater(t4["cross_vendor_workload_portability_pct"], 95.0)
        self.assertLess(
            t4["monthly_25b_tokens_fleet_tco_usd"],
            t1["monthly_25b_tokens_fleet_tco_usd"] * 0.45,
        )

    def test_artifacts_exist(self) -> None:
        slug = "multi-vendor-ai-accelerators-nvidia-tpu-amd-mi325x-trainium2"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

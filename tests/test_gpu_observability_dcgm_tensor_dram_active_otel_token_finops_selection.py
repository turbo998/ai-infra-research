import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGpuObservabilityDcgmOtelTokenFinopsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "gpu_observability_dcgm_tensor_dram_active_otel_token_finops_selection"
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
        t1 = tiers["tier1_nvidia_smi_gpu_util_only_and_flat_gpu_hour_chargeback"]
        t4 = tiers["tier4_dcgm_4quadrant_profiling_plus_otel_genai_weighted_token_finops"]
        self.assertGreater(t4["true_hardware_efficiency_mbu_mfu_pct"], 75.0)
        self.assertLess(t4["unattributed_or_wasted_gpu_spend_pct"], 2.5)
        self.assertLess(
            t4["monthly_256gpu_cluster_net_tco_usd"],
            t1["monthly_256gpu_cluster_net_tco_usd"] * 0.45,
        )

    def test_artifacts_exist(self) -> None:
        slug = "gpu-observability-dcgm-tensor-dram-active-otel-token-finops"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

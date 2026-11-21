import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGoogleCloudTpuVsNvidiaGpuSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection"
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
        t1 = tiers["tier1_unmodified_dynamic_shape_pytorch_cuda_on_tpu_nonwrap"]
        t2 = tiers["tier2_homogeneous_h200_rocev2_for_all_workloads"]
        t4 = tiers["tier4_workload_aligned_tpu7x_ironwood_v6e_8t_8i_plus_gpu_hybrid"]
        self.assertGreater(t4["sustained_cluster_mfu_pct"], 57.0)
        self.assertGreater(t4["cluster_ocs_or_fabric_goodput_pct"], 99.0)
        self.assertLess(t4["serving_cost_per_1m_tokens_usd"], 0.28)
        self.assertLess(
            t4["monthly_2048_chip_equivalent_tco_usd"],
            t2["monthly_2048_chip_equivalent_tco_usd"] * 0.65,
        )
        self.assertGreater(t4["sustained_cluster_mfu_pct"], t1["sustained_cluster_mfu_pct"] * 3.5)

    def test_artifacts_exist(self) -> None:
        slug = "google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

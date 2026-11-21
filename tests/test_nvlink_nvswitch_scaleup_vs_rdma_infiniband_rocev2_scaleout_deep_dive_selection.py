import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestNvlinkVsRdmaIbRocev2DeepDiveSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "nvlink_nvswitch_scaleup_vs_rdma_infiniband_rocev2_scaleout_deep_dive_selection"
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
        t1 = tiers["tier1_rocev2_without_peermem_or_ecn_static_ecmp"]
        t4 = tiers["tier4_hierarchical_nvlink_nvls_scaleup_plus_tuned_rdma_scaleout"]
        self.assertGreater(t4["nccl_allreduce_busbw_gb_s"], 45.0)
        self.assertEqual(t4["monthly_pfc_deadlock_and_straggler_incidents"], 0)
        self.assertLess(
            t4["monthly_512gpu_network_and_wasted_compute_tco_usd"],
            t1["monthly_512gpu_network_and_wasted_compute_tco_usd"] * 0.35,
        )

    def test_artifacts_exist(self) -> None:
        slug = "nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAiHypercomputerGkeKueueSlurmStorageSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection"
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
        t1 = tiers["tier1_vanilla_k8s_spot_uncached_gcs_fuse_sync_ckpt"]
        t4 = tiers["tier4_gke_kueue_tas_jobset_dws_hyperdisk_ml_orbax_async"]
        self.assertGreater(t4["end_to_end_cluster_goodput_pct"], 96.0)
        self.assertLess(t4["checkpoint_stall_seconds"], 3.0)
        self.assertEqual(t4["partial_allocation_deadlock_waste_pct"], 0.0)
        self.assertLess(
            t4["monthly_1024_chip_compute_and_storage_tco_usd"],
            t1["monthly_1024_chip_compute_and_storage_tco_usd"] * 0.45,
        )

    def test_artifacts_exist(self) -> None:
        slug = "ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

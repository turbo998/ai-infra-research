import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAiStorage3fsLustreWekaVastGdsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "ai_storage_3fs_lustre_weka_vast_gds_checkpoint_kv_io_selection"
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
        t1 = tiers["tier1_standard_nfs_or_uncached_object_fuse_with_small_files"]
        t4 = tiers["tier4_converged_rdma_nvme_3fs_plus_pinned_async_ckpt_and_gds_kv_pool"]
        self.assertGreater(t4["sustained_cluster_read_bw_gb_s"], 2000.0)
        self.assertLess(t4["synchronous_checkpoint_stall_sec"], 3.0)
        self.assertLess(
            t4["monthly_512gpu_storage_and_io_stall_tco_usd"],
            t1["monthly_512gpu_storage_and_io_stall_tco_usd"] * 0.20,
        )

    def test_artifacts_exist(self) -> None:
        slug = "ai-storage-3fs-lustre-weka-vast-gds-checkpoint-kv-io"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

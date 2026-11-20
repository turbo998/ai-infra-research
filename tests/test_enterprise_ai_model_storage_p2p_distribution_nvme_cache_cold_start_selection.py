import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiModelStorageColdStartSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_model_storage_p2p_distribution_nvme_cache_cold_start_selection"
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
        t1 = tiers["tier1_direct_s3_blob_http_pull"]
        t4 = tiers["tier4_s3_dragonfly_p2p_local_nvme_safetensors_streamer"]
        self.assertLessEqual(t4["thirty_two_node_cold_start_sec"], 15.0)
        self.assertGreater(t1["thirty_two_node_cold_start_sec"], 900.0)
        self.assertLess(
            t4["monthly_storage_and_idle_gpu_buffer_tco_usd"],
            t1["monthly_storage_and_idle_gpu_buffer_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-ai-model-storage-p2p-distribution-nvme-cache-cold-start"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

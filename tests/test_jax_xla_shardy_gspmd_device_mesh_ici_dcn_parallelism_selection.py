import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestJaxXlaShardyGspmdDeviceMeshSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "jax_xla_shardy_gspmd_device_mesh_ici_dcn_parallelism_selection"
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
        t1 = tiers["tier1_misaligned_mesh_tp_or_fsdp_across_dcn"]
        t3 = tiers["tier3_legacy_gspmd_on_hybrid_moe_multislice"]
        t4 = tiers["tier4_shardy_mlir_hierarchical_mesh_shard_map_ici_dcn_aligned"]
        self.assertGreater(t4["sustained_cluster_mfu_pct"], 59.0)
        self.assertGreater(t4["sustained_cluster_mfu_pct"] - t3["sustained_cluster_mfu_pct"], 10.0)
        self.assertEqual(t4["redundant_reshard_bytes_per_step_gb"], 0.0)
        self.assertLess(
            t4["monthly_2048_chip_training_tco_usd"],
            t1["monthly_2048_chip_training_tco_usd"] * 0.40,
        )

    def test_artifacts_exist(self) -> None:
        slug = "jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

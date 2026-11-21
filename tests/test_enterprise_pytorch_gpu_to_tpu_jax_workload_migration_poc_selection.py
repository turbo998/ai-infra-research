import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterprisePytorchToTpuJaxMigrationSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection"
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
        t1 = tiers["tier1_blind_1to1_eager_pytorch_xla_port_of_all_workloads"]
        t4 = tiers["tier4_3lane_triage_maxtext_vllm_torchax_14day_poc_gate"]
        self.assertGreater(t4["poc_success_and_sla_pass_rate_pct"], 98.0)
        self.assertGreater(t4["post_migration_sustained_mfu_pct"], 57.0)
        self.assertLess(t4["avg_migration_engineering_weeks_per_model"], 1.5)
        self.assertLess(
            t4["monthly_fleet_compute_and_engineering_tco_usd"],
            t1["monthly_fleet_compute_and_engineering_tco_usd"] * 0.50,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestGpuSharingMigMpsTimeslicingTidalSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "gpu_sharing_mig_mps_timeslicing_hami_tidal_colocation_selection"
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
        t1 = tiers["tier1_whole_gpu_per_small_model_and_zero_night_colocation"]
        t4 = tiers["tier4_mig_mps_hybrid_plus_kueue_tidal_colocation_and_vllm_sleep_wake"]
        self.assertGreater(t4["effective_24h_gpu_compute_and_hbm_utilization_pct"], 85.0)
        self.assertEqual(t4["noisy_neighbor_oom_or_p99_sla_breach_rate_pct"], 0.0)
        self.assertLess(
            t4["monthly_64gpu_platform_tco_usd"],
            t1["monthly_64gpu_platform_tco_usd"] * 0.35,
        )

    def test_artifacts_exist(self) -> None:
        slug = "gpu-sharing-mig-mps-timeslicing-hami-tidal-colocation"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

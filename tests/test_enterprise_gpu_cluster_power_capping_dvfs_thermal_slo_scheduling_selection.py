import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseGpuClusterPowerCappingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection"
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
        t1 = tiers["tier1_uncapped_700w_nameplate_provisioning"]
        t4 = tiers["tier4_phase_aware_dvfs_600w_prefill_450w_decode_pdu_clamp"]
        self.assertGreater(t4["slo_goodput_retention_pct"], 98.0)
        self.assertGreater(t4["gpus_packed_per_32kw_rack"], t1["gpus_packed_per_32kw_rack"])
        self.assertGreaterEqual(t4["cluster_power_reduction_pct"], 25.0)
        self.assertLess(
            t4["monthly_colo_rack_and_power_tco_usd"],
            t1["monthly_colo_rack_and_power_tco_usd"] * 0.65,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

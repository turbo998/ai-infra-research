import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseMultiTenantMaasPlatformQosVtcSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_multi_tenant_maas_platform_qos_vtc_noisy_neighbor_selection"
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
        t1 = tiers["tier1_single_fifo_fcfs_shared_pool_without_tenant_fair_queueing"]
        t4 = tiers["tier4_3tier_qos_plus_vtc_token_fair_queueing_and_work_conserving_quota_borrowing"]
        self.assertLess(t4["gold_tenant_p99_ttft_under_burst_ms"], 450.0)
        self.assertEqual(t4["noisy_neighbor_sla_violation_rate_pct"], 0.0)
        self.assertGreater(t4["cluster_overall_gpu_utilization_pct"], 85.0)
        self.assertLess(t4["monthly_64gpu_maas_platform_tco_usd"], t1["monthly_64gpu_maas_platform_tco_usd"] * 0.50)

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

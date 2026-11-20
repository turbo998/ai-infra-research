import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiVendorLockInSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_vendor_lock_in_exit_strategy_open_standards_portability_selection"
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
        t1 = tiers["tier1_proprietary_stateful_sdk_lock_in"]
        t4 = tiers["tier4_four_layer_open_standards_portability"]
        self.assertEqual(t4["app_code_rewrite_fraction_pct"], 0.0)
        self.assertLessEqual(t4["provider_migration_lead_time_days"], 3.0)
        self.assertGreaterEqual(t4["native_frontier_feature_retention_pct"], 98.0)
        self.assertLess(t4["provider_switch_total_cost_usd"], t1["provider_switch_total_cost_usd"] * 0.05)

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-ai-vendor-lock-in-exit-strategy-open-standards-portability"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").svg if False else (self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

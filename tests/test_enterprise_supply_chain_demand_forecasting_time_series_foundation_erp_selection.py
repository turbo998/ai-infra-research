import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseSupplyChainDemandForecastingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection"
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
        t1 = tiers["tier1_direct_text_llm_csv_prompt_forecasting"]
        t4 = tiers["tier4_lightgbm_chronos_tsfm_llm_covariates_conformal_ortools"]
        self.assertLess(t4["overall_wape_error_pct"], 10.0)
        self.assertLess(t4["cold_start_sku_wape_error_pct"], 14.0)
        self.assertLess(t4["shelf_stockout_rate_pct"], 1.5)
        self.assertLess(
            t4["monthly_inventory_stockout_and_compute_tco_usd"],
            t1["monthly_inventory_stockout_and_compute_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-supply-chain-demand-forecasting-time-series-foundation-erp"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

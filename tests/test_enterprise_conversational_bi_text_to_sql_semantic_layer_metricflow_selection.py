import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseConversationalBiSemanticLayerSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection"
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
        t1 = tiers["tier1_raw_ddl_freeform_text_to_sql"]
        t4 = tiers["tier4_semantic_layer_metricflow_ast_dryrun_rls"]
        self.assertGreater(t4["executive_metric_accuracy_pct"], 98.0)
        self.assertEqual(t4["runaway_full_table_scan_rate_pct"], 0.0)
        self.assertEqual(t4["rls_unauthorized_exposure_risk_pct"], 0.0)
        self.assertLess(
            t4["monthly_warehouse_and_rework_tco_usd"],
            t1["monthly_warehouse_and_rework_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()

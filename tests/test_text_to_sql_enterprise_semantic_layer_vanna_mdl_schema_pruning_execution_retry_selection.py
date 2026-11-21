import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestTextToSQLEnterpriseSemanticLayerMDLSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "text_to_sql_enterprise_semantic_layer_vanna_mdl_schema_pruning_execution_retry_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "text_to_sql_enterprise_semantic_layer_vanna_mdl_schema_pruning_execution_retry_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_Semantic_MDL_Column_Value_Pruning_SQLGlot_AST_Gate"]["Business_Logic_Execution_Accuracy_Pct"],
            tiers["Tier_1_Full_Raw_DDL_96K_Context_Dump"]["Business_Logic_Execution_Accuracy_Pct"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "text-to-sql-enterprise-semantic-layer-vanna-mdl-schema-pruning-execution-retry-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "text-to-sql-enterprise-semantic-layer-vanna-mdl-schema-pruning-execution-retry-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "text-to-sql-enterprise-semantic-layer-vanna-mdl-schema-pruning-execution-retry-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "text-to-sql-enterprise-semantic-layer-vanna-mdl-schema-pruning-execution-retry-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "text-to-sql-enterprise-semantic-layer-vanna-mdl-schema-pruning-execution-retry-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
